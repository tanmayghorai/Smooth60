// Smooth60 — capture-card viewer for Mac that turns 30 fps console games into 60 fps
// using Apple's built-in VideoToolbox low-latency frame interpolation (Apple Silicon, macOS 26+).
//
// Pipeline:
//   capture thread (60 Hz): drop the repeated frames a 30 fps game produces, copy new ones
//   processing thread: generate in-between frame(s) with VTFrameProcessor (1 → 60 fps, 3 → 120 fps)
//   playback clock: every frame is shown a fixed delay after it was captured, evenly spaced,
//   with the game audio on the same clock so they stay in sync.

import AppKit
import IOKit
import AVFoundation
import CoreMedia
import CoreVideo
import QuartzCore
import CoreImage
import CoreAudio
import VideoToolbox
import Metal
import MetalFX

// MARK: - Duplicate frame detection

/// A 30 fps game inside a 60 Hz HDMI signal arrives as A A B B C C …
/// This compares a sparse grid of brightness samples with the previous frame.
final class DuplicateDetector {
    private var previous = [UInt8]()
    private var current = [UInt8]()
    private let gridX = 160
    private let gridY = 90

    func isDuplicate(_ pixelBuffer: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let planar = CVPixelBufferIsPlanar(pixelBuffer)
        let baseAddress = planar ? CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
                                 : CVPixelBufferGetBaseAddress(pixelBuffer)
        guard let base = baseAddress else { return false }
        let bytesPerRow = planar ? CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
                                 : CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = planar ? CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
                           : CVPixelBufferGetWidth(pixelBuffer)
        let height = planar ? CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
                            : CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = planar ? min(width, bytesPerRow) : bytesPerRow
        guard height > 0, rowBytes > 0 else { return false }

        let pixels = base.assumingMemoryBound(to: UInt8.self)
        current.removeAll(keepingCapacity: true)
        for j in 0..<gridY {
            let y = min(height - 1, (j * height + height / 2) / gridY)
            let row = pixels + y * bytesPerRow
            for i in 0..<gridX {
                let x = min(rowBytes - 1, (i * rowBytes + rowBytes / 2) / gridX)
                current.append(row[x])
            }
        }
        defer { swap(&previous, &current) }
        guard previous.count == current.count else { return false }

        var changed = 0
        var total = 0
        for k in 0..<current.count {
            let d = abs(Int(current[k]) - Int(previous[k]))
            total += d
            if d > 10 { changed += 1 }
        }
        let mean = Double(total) / Double(current.count)
        return changed < 3 && mean < 0.6
    }
}

/// Brightness samples on a grid — used to sanity-check generated frames.
func lumaSamples(_ pixelBuffer: CVPixelBuffer, gridX: Int = 96, gridY: Int = 54) -> [UInt8] {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    let planar = CVPixelBufferIsPlanar(pixelBuffer)
    guard let base = planar ? CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
                            : CVPixelBufferGetBaseAddress(pixelBuffer) else { return [] }
    let bytesPerRow = planar ? CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0) : CVPixelBufferGetBytesPerRow(pixelBuffer)
    let width = planar ? CVPixelBufferGetWidthOfPlane(pixelBuffer, 0) : CVPixelBufferGetWidth(pixelBuffer)
    let height = planar ? CVPixelBufferGetHeightOfPlane(pixelBuffer, 0) : CVPixelBufferGetHeight(pixelBuffer)
    let rowBytes = planar ? min(width, bytesPerRow) : bytesPerRow
    guard height > 0, rowBytes > 0 else { return [] }
    let pixels = base.assumingMemoryBound(to: UInt8.self)
    var out = [UInt8]()
    out.reserveCapacity(gridX * gridY)
    for j in 0..<gridY {
        let row = pixels + min(height - 1, (j * height + height / 2) / gridY) * bytesPerRow
        for i in 0..<gridX { out.append(row[min(rowBytes - 1, (i * rowBytes + rowBytes / 2) / gridX)]) }
    }
    return out
}

func meanDifference(_ a: [UInt8], _ b: [UInt8]) -> Double {
    guard a.count == b.count, !a.isEmpty else { return -1 }
    var total = 0
    for k in 0..<a.count { total += abs(Int(a[k]) - Int(b[k])) }
    return Double(total) / Double(a.count)
}

// MARK: - Frame generation (Apple VideoToolbox)

enum GeneratorError: Error, CustomStringConvertible {
    case unsupported, configFailed, poolFailed
    var description: String {
        switch self {
        case .unsupported: return "This Mac doesn't support low-latency frame interpolation"
        case .configFailed: return "Couldn't configure frame interpolation for this resolution"
        case .poolFailed: return "Couldn't allocate frame buffers"
        }
    }
}

final class FrameGenerator {
    let width: Int
    let height: Int
    private let processor = VTFrameProcessor()
    private var sourcePool: CVPixelBufferPool?
    private var destinationPool: CVPixelBufferPool?
    private var transferSession: VTPixelTransferSession?
    private(set) var lastError: String?
    private(set) var sizeRejections = 0
    /// 1 → 30 fps becomes 60. 3 → 30 fps becomes 120.
    let framesPerGap: Int

    init(width: Int, height: Int, framesPerGap: Int) throws {
        self.width = width
        self.height = height
        self.framesPerGap = framesPerGap == 3 ? 3 : 1
        guard VTLowLatencyFrameInterpolationConfiguration.isSupported else { throw GeneratorError.unsupported }
        let maybeConfig: VTLowLatencyFrameInterpolationConfiguration? =
            VTLowLatencyFrameInterpolationConfiguration(frameWidth: width,
                                                        frameHeight: height,
                                                        numberOfInterpolatedFrames: self.framesPerGap)
        guard let config = maybeConfig else { throw GeneratorError.configFailed }
        try processor.startSession(configuration: config)

        sourcePool = FrameGenerator.makePool(config.sourcePixelBufferAttributes, width: width, height: height)
        destinationPool = FrameGenerator.makePool(config.destinationPixelBufferAttributes, width: width, height: height)
        guard sourcePool != nil, destinationPool != nil else { throw GeneratorError.poolFailed }

        var session: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session)
        transferSession = session
        guard transferSession != nil else { throw GeneratorError.poolFailed }
    }

    deinit {
        processor.endSession()
        if let transferSession { VTPixelTransferSessionInvalidate(transferSession) }
    }

    static func makePool(_ rawAttributes: Any?, width: Int, height: Int) -> CVPixelBufferPool? {
        var attributes = (rawAttributes as? [String: Any]) ?? [:]
        attributes[kCVPixelBufferWidthKey as String] = width
        attributes[kCVPixelBufferHeightKey as String] = height
        if attributes[kCVPixelBufferPixelFormatTypeKey as String] == nil {
            attributes[kCVPixelBufferPixelFormatTypeKey as String] = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        }
        if attributes[kCVPixelBufferIOSurfacePropertiesKey as String] == nil {
            attributes[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]()
        }
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        return pool
    }

    /// Copies a capture-card frame into a buffer in the format the interpolator wants.
    func makeSourceCopy(of raw: CVPixelBuffer) -> CVPixelBuffer? {
        guard let pool = sourcePool, let session = transferSession else { return nil }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out) == kCVReturnSuccess,
              let copy = out else { return nil }
        let status = VTPixelTransferSessionTransferImage(session, from: raw, to: copy)
        guard status == noErr else { lastError = "Pixel transfer failed (\(status))"; return nil }
        CVBufferPropagateAttachments(raw, copy)
        return copy
    }

    /// Time of in-between frame `index` (0-based) out of `framesPerGap`, evenly spaced.
    static func time(_ index: Int, of count: Int, from start: CMTime, to end: CMTime) -> CMTime {
        CMTimeAdd(start, CMTimeMultiplyByRatio(CMTimeSubtract(end, start),
                                               multiplier: Int32(index + 1), divisor: Int32(count + 1)))
    }

    /// Generates the evenly spaced in-between frames (1 or 3) between `previous` and `current`.
    func interpolate(previous: CVPixelBuffer, previousTime: CMTime,
                     current: CVPixelBuffer, currentTime: CMTime) -> [CVPixelBuffer]? {
        guard let pool = destinationPool else { return nil }
        var destinations = [CVPixelBuffer]()
        var destinationFrames = [VTFrameProcessorFrame]()
        for i in 0..<framesPerGap {
            var out: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out) == kCVReturnSuccess,
                  let destination = out else { return nil }
            let t = FrameGenerator.time(i, of: framesPerGap, from: previousTime, to: currentTime)
            let frame: VTFrameProcessorFrame? = VTFrameProcessorFrame(buffer: destination, presentationTimeStamp: t)
            guard let frame else { lastError = "Couldn't wrap frames"; return nil }
            destinations.append(destination)
            destinationFrames.append(frame)
        }

        let previousFrame: VTFrameProcessorFrame? = VTFrameProcessorFrame(buffer: previous, presentationTimeStamp: previousTime)
        let currentFrame: VTFrameProcessorFrame? = VTFrameProcessorFrame(buffer: current, presentationTimeStamp: currentTime)
        guard let previousFrame, let currentFrame else {
            lastError = "Couldn't wrap frames"
            return nil
        }

        let maybeParameters: VTLowLatencyFrameInterpolationParameters?
        if framesPerGap == 3 {
            maybeParameters = VTLowLatencyFrameInterpolationParameters(sourceFrame: currentFrame,
                                                                       previousFrame: previousFrame,
                                                                       interpolationPhase: [0.25, 0.5, 0.75],
                                                                       destinationFrames: destinationFrames)
        } else {
            maybeParameters = VTLowLatencyFrameInterpolationParameters(sourceFrame: currentFrame,
                                                                       previousFrame: previousFrame,
                                                                       interpolationPhase: [0.5],
                                                                       destinationFrames: destinationFrames)
        }
        guard let parameters = maybeParameters else {
            lastError = "Couldn't create interpolation parameters"
            return nil
        }
        // Swift only exposes the asynchronous call, so wait for it to finish (we're on the capture queue).
        final class Outcome: @unchecked Sendable { var error: Error? }
        let outcome = Outcome()
        let done = DispatchSemaphore(value: 0)
        processor.process(parameters: parameters, completionHandler: { _, error in
            outcome.error = error
            done.signal()
        })
        if done.wait(timeout: .now() + .milliseconds(100)) == .timedOut {
            lastError = "Interpolation timed out"
            return nil
        }
        if let error = outcome.error {
            lastError = "Interpolation failed: \(error.localizedDescription)"
            // -19730 "Processor is not initialized": this frame size is too big for this Mac.
            if (error as NSError).code == -19730 { sizeRejections += 1 }
            return nil
        }
        sizeRejections = 0
        for destination in destinations { CVBufferPropagateAttachments(current, destination) }
        return destinations
    }
}

// MARK: - Upscaling (MetalFX Spatial — Apple's real-time game upscaler)

final class Upscaler {
    let inputWidth: Int
    let inputHeight: Int
    let targetHeight: Int
    let outputWidth: Int
    let outputHeight: Int
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let scaler: MTLFXSpatialScaler
    private let scaledTexture: MTLTexture
    private var textureCache: CVMetalTextureCache?
    private var inputPool: CVPixelBufferPool?
    private var outputPool: CVPixelBufferPool?
    private var transfer: VTPixelTransferSession?
    private(set) var lastError: String?

    static var isAvailable: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return MTLFXSpatialScalerDescriptor.supportsDevice(device)
    }

    /// Output heights offered for a given input height (1080p in → 1440p or 4K out).
    static func targets(forInputHeight height: Int) -> [Int] {
        isAvailable ? [1440, 2160].filter { $0 > height } : []
    }

    static func outputSize(width: Int, height: Int, targetHeight: Int) -> (Int, Int) {
        let w = (width * targetHeight + height / 2) / height
        return (w + (w & 1), targetHeight)
    }

    private static func bgraPool(width: Int, height: Int) -> CVPixelBufferPool? {
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        return pool
    }

    init(width: Int, height: Int, targetHeight: Int) throws {
        inputWidth = width
        inputHeight = height
        self.targetHeight = targetHeight
        (outputWidth, outputHeight) = Upscaler.outputSize(width: width, height: height, targetHeight: targetHeight)
        guard let device = MTLCreateSystemDefaultDevice(), MTLFXSpatialScalerDescriptor.supportsDevice(device),
              let queue = device.makeCommandQueue() else { throw GeneratorError.unsupported }
        self.device = device
        commandQueue = queue

        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = width
        descriptor.inputHeight = height
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.colorProcessingMode = .perceptual
        guard let scaler = descriptor.makeSpatialScaler(device: device) else { throw GeneratorError.configFailed }
        scaler.inputContentWidth = width
        scaler.inputContentHeight = height
        self.scaler = scaler

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                         width: outputWidth, height: outputHeight,
                                                                         mipmapped: false)
        textureDescriptor.usage = scaler.outputTextureUsage
        textureDescriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else { throw GeneratorError.poolFailed }
        scaledTexture = texture

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        inputPool = Upscaler.bgraPool(width: width, height: height)
        outputPool = Upscaler.bgraPool(width: outputWidth, height: outputHeight)
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
        guard textureCache != nil, inputPool != nil, outputPool != nil, transfer != nil else { throw GeneratorError.poolFailed }
    }

    deinit {
        if let transfer { VTPixelTransferSessionInvalidate(transfer) }
    }

    private func texture(for pixelBuffer: CVPixelBuffer, usage: MTLTextureUsage) -> (MTLTexture, CVMetalTexture)? {
        guard let textureCache else { return nil }
        var cvTexture: CVMetalTexture?
        let attributes = [kCVMetalTextureUsage as String: NSNumber(value: usage.rawValue)] as CFDictionary
        guard CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache, pixelBuffer, attributes,
                                                        .bgra8Unorm, CVPixelBufferGetWidth(pixelBuffer),
                                                        CVPixelBufferGetHeight(pixelBuffer), 0, &cvTexture) == kCVReturnSuccess,
              let cvTexture, let texture = CVMetalTextureGetTexture(cvTexture) else { return nil }
        return (texture, cvTexture)
    }

    func upscale(_ input: CVPixelBuffer, at time: CMTime) -> CVPixelBuffer? {
        guard let inputPool, let outputPool, let transfer else { return nil }
        var rgbOut: CVPixelBuffer?
        var bigOut: CVPixelBuffer?
        // 1. YUV → RGB at the input size.
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, inputPool, &rgbOut) == kCVReturnSuccess,
              let rgb = rgbOut,
              VTPixelTransferSessionTransferImage(transfer, from: input, to: rgb) == noErr,
              CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outputPool, &bigOut) == kCVReturnSuccess,
              let big = bigOut else { lastError = "Couldn't prepare frame"; return nil }
        // 2. Upscale on the GPU, then copy into a frame the display and recorder can use.
        guard let (inputTexture, inputRef) = texture(for: rgb, usage: scaler.colorTextureUsage),
              let (outputTexture, outputRef) = texture(for: big, usage: [.shaderRead, .shaderWrite]),
              let commandBuffer = commandQueue.makeCommandBuffer() else { lastError = "Couldn't set up the GPU"; return nil }
        scaler.colorTexture = inputTexture
        scaler.outputTexture = scaledTexture
        scaler.encode(commandBuffer: commandBuffer)
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: scaledTexture, to: outputTexture)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        _ = (inputRef, outputRef)   // keep the texture wrappers alive until the GPU is done
        if let error = commandBuffer.error { lastError = "Upscaling failed: \(error.localizedDescription)"; return nil }
        CVBufferPropagateAttachments(input, big)
        return big
    }
}

// MARK: - Enhancement (Apple temporal noise filter + Core Image sharpen / picture boost)

enum EnhanceLevel: Int {
    case off = 0, light, medium, strong
    var title: String { ["Off", "Light  (sharpen + colour)", "Medium  (clean-up + sharpen + colour)", "Strong  (strong clean-up + sharpen + colour)"][rawValue] }
    var name: String { ["Off", "Light", "Medium", "Strong"][rawValue] }
    var denoiseStrength: Float { [0, 0, 0.5, 1.0][rawValue] }
    var sharpness: Double { [0, 0.25, 0.4, 0.6][rawValue] }
    var vibrance: Double { [0, 0.10, 0.18, 0.26][rawValue] }
    var contrast: Double { [1, 1.02, 1.035, 1.05][rawValue] }
}

/// How tall the protected subtitle strip at the bottom is.
enum SubtitleMode: Int {
    case auto = 0, normal, large, largest
    var title: String { ["Auto (finds captions)", "Normal (13%)", "Large (18%)", "Largest (24%)"][rawValue] }
    var fixedHeight: CGFloat { [0.13, 0.13, 0.18, 0.24][rawValue] }
}

final class Enhancer {
    let width: Int
    let height: Int
    let format: OSType
    let level: EnhanceLevel
    let protectHUD: Bool
    let subtitleMode: SubtitleMode
    /// The subtitle strip as fractions of the frame: top (from the bottom edge), left, right.
    private(set) var band = (top: CGFloat(0.13), left: CGFloat(0.2), right: CGFloat(0.8))
    private var bandHold = 0                       // frames to keep a grown strip before shrinking it
    private var detectCounter = 0
    private var masks: [String: CVPixelBuffer] = [:]
    var denoiseAllowed = true              // set false if the Mac can't keep up
    private(set) var denoiseFailed: String?
    private(set) var lastError: String?
    private(set) var denoiseMs = 0.0, denoiseCount = 0
    private(set) var pictureMs = 0.0, pictureCount = 0

    private var processor: VTFrameProcessor?
    private var noiseConfig: VTTemporalNoiseFilterConfiguration?
    private var noiseSourcePool: CVPixelBufferPool?
    private var noiseDestinationPool: CVPixelBufferPool?
    private var plainPool: CVPixelBufferPool?
    private var outputPool: CVPixelBufferPool?
    private var transfer: VTPixelTransferSession?
    private var previous: (buffer: CVPixelBuffer, time: CMTime)?
    private var usePreviousFrame = true
    private let context: CIContext

    var denoiseActive: Bool { level.denoiseStrength > 0 && denoiseAllowed && denoiseFailed == nil }

    private static func pool(_ attributes: [String: Any]?, width: Int, height: Int, format: OSType) -> CVPixelBufferPool? {
        var a = attributes ?? [:]
        a[kCVPixelBufferWidthKey as String] = width
        a[kCVPixelBufferHeightKey as String] = height
        if a[kCVPixelBufferPixelFormatTypeKey as String] == nil { a[kCVPixelBufferPixelFormatTypeKey as String] = format }
        if a[kCVPixelBufferIOSurfacePropertiesKey as String] == nil { a[kCVPixelBufferIOSurfacePropertiesKey as String] = [String: Any]() }
        a[kCVPixelBufferMetalCompatibilityKey as String] = true
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, a as CFDictionary, &pool)
        return pool
    }

    init(width: Int, height: Int, format: OSType, level: EnhanceLevel, protectHUD: Bool, subtitleMode: SubtitleMode = .auto) {
        self.width = width
        self.height = height
        self.format = format
        self.level = level
        self.protectHUD = protectHUD
        self.subtitleMode = subtitleMode
        band = (subtitleMode.fixedHeight, 0.2, 0.8)
        if let device = MTLCreateSystemDefaultDevice() {
            context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false, .name: "Smooth60 enhance",
                                                             .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        } else {
            context = CIContext(options: [.cacheIntermediates: false, .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        }
        VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer)
        outputPool = Enhancer.pool(nil, width: width, height: height, format: format)
        if level.denoiseStrength > 0 { setUpDenoise() }
    }

    deinit {
        processor?.endSession()
        if let transfer { VTPixelTransferSessionInvalidate(transfer) }
    }

    private func setUpDenoise() {
        guard VTTemporalNoiseFilterConfiguration.isSupported else { denoiseFailed = "not supported on this Mac"; return }
        // Apple's filter only takes its own compressed formats: '&8v0' (video range) / '&8f0' (full range).
        let noiseFormat: OSType = format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 0x2638_6630 : 0x2638_7630
        let maybeConfig: VTTemporalNoiseFilterConfiguration? =
            VTTemporalNoiseFilterConfiguration(frameWidth: width, frameHeight: height, sourcePixelFormat: noiseFormat)
        guard let config = maybeConfig else { denoiseFailed = "can't configure for \(width)×\(height)"; return }
        let processor = VTFrameProcessor()
        do { try processor.startSession(configuration: config) } catch { denoiseFailed = "\(error.localizedDescription)"; return }
        noiseSourcePool = Enhancer.pool(config.sourcePixelBufferAttributes, width: width, height: height, format: noiseFormat)
        noiseDestinationPool = Enhancer.pool(config.destinationPixelBufferAttributes, width: width, height: height, format: noiseFormat)
        plainPool = Enhancer.pool(nil, width: width, height: height, format: format)
        guard noiseSourcePool != nil, noiseDestinationPool != nil, plainPool != nil else {
            processor.endSession()
            denoiseFailed = "couldn't allocate buffers"
            return
        }
        self.processor = processor
        noiseConfig = config
    }

    private func newBuffer(_ pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        return buffer
    }

    private func runFilter(source: CVPixelBuffer, destination: CVPixelBuffer, time: CMTime,
                           previousFrame: (buffer: CVPixelBuffer, time: CMTime)?) -> Error? {
        guard let processor,
              let src = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: time),
              let dst = VTFrameProcessorFrame(buffer: destination, presentationTimeStamp: time) else { return GeneratorError.poolFailed }
        var previousFrames: [VTFrameProcessorFrame] = []
        if let previousFrame, let f = VTFrameProcessorFrame(buffer: previousFrame.buffer, presentationTimeStamp: previousFrame.time) {
            previousFrames.append(f)
        }
        let maybeParams: VTTemporalNoiseFilterParameters? =
            VTTemporalNoiseFilterParameters(sourceFrame: src, nextFrames: [], previousFrames: previousFrames,
                                            destinationFrame: dst, filterStrength: level.denoiseStrength,
                                            hasDiscontinuity: previousFrames.isEmpty)
        guard let params = maybeParams else { return GeneratorError.configFailed }
        final class Box: @unchecked Sendable { var error: Error? }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        processor.process(parameters: params, completionHandler: { _, error in
            box.error = error
            done.signal()
        })
        if done.wait(timeout: .now() + 0.1) == .timedOut { return GeneratorError.configFailed }
        return box.error
    }

    /// Clean-up pass. Returns a plain (non-compressed) frame, or nil to use the original.
    private func denoise(_ frame: CVPixelBuffer, at time: CMTime) -> CVPixelBuffer? {
        guard let transfer, let source = newBuffer(noiseSourcePool), let destination = newBuffer(noiseDestinationPool),
              let plain = newBuffer(plainPool),
              VTPixelTransferSessionTransferImage(transfer, from: frame, to: source) == noErr else {
            lastError = "clean-up: couldn't prepare frame"
            return nil
        }
        var prior = usePreviousFrame ? previous : nil
        if let p = prior, CMTimeGetSeconds(CMTimeSubtract(time, p.time)) > 0.2 || CMTimeCompare(p.time, time) >= 0 { prior = nil }
        var error = runFilter(source: source, destination: destination, time: time, previousFrame: prior)
        if error != nil && prior != nil {
            usePreviousFrame = false           // this Mac wants it without a reference frame
            error = runFilter(source: source, destination: destination, time: time, previousFrame: nil)
        }
        previous = (source, time)
        if let error {
            lastError = "clean-up: \(error.localizedDescription)"
            return nil
        }
        guard VTPixelTransferSessionTransferImage(transfer, from: destination, to: plain) == noErr else {
            lastError = "clean-up: couldn't convert result"
            return nil
        }
        CVBufferPropagateAttachments(frame, plain)
        Enhancer.tagHD(plain)
        return plain
    }

    /// Screen areas left untouched when HUD protection is on: the four corners
    /// (health, money, minimap, prompts) and the subtitle band at the bottom.
    /// Protected areas as fractions of the frame (origin bottom-left): four corners + the subtitle strip.
    var protectedRects: [CGRect] {
        [
            CGRect(x: 0, y: 0, width: 0.24, height: 0.20),
            CGRect(x: 0.76, y: 0, width: 0.24, height: 0.20),
            CGRect(x: 0, y: 0.80, width: 0.24, height: 0.20),
            CGRect(x: 0.76, y: 0.80, width: 0.24, height: 0.20),
            CGRect(x: band.left, y: 0, width: band.right - band.left, height: band.top),
        ]
    }

    private func hudRects() -> [CGRect] {
        let w = CGFloat(width), h = CGFloat(height)
        return protectedRects.map { CGRect(x: ($0.minX * w).rounded(), y: ($0.minY * h).rounded(),
                                           width: ($0.width * w).rounded(), height: ($0.height * h).rounded()) }
    }

    /// Finds on-screen captions near the bottom: rows with many sharp bright-to-dark edges (white text
    /// with a dark outline or shadow), stacked upwards from the subtitle area. Returns how far up they
    /// reach and how wide they are, as fractions, or nil when there's no caption.
    private func findCaptions(in frame: CVPixelBuffer) -> (top: CGFloat, left: CGFloat, right: CGFloat)? {
        guard CVPixelBufferGetPlaneCount(frame) >= 1 else { return nil }
        CVPixelBufferLockBaseAddress(frame, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(frame, 0) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(frame, 0)
        let w = CVPixelBufferGetWidthOfPlane(frame, 0), h = CVPixelBufferGetHeightOfPlane(frame, 0)
        let luma = base.assumingMemoryBound(to: UInt8.self)
        let step = max(2, w / 480)                  // ~4 px at 1080p
        let rowStep = max(1, h / 540)               // ~2 px at 1080p
        let x0 = w / 10, x1 = w - w / 10            // skip the corner HUD
        let searchRows = Int(Double(h) * 0.34)      // never look above the bottom third
        let lineGap = Int(Double(h) * 0.05)         // max gap between stacked caption lines
        var reach = Int(Double(h) * 0.13)           // rows up from the bottom edge
        var found = false
        var minX = w, maxX = 0
        var up = 0
        while up < searchRows {
            let row = luma + (h - 1 - up) * rowBytes
            var edges = 0, rowMin = w, rowMax = 0
            var x = x0
            while x + step < x1 {
                let a = Int(row[x]), b = Int(row[x + step])
                if max(a, b) > 175 && abs(a - b) > 70 {
                    edges += 1
                    if x < rowMin { rowMin = x }
                    if x > rowMax { rowMax = x }
                }
                x += step
            }
            if edges >= 8 && up <= reach + lineGap {   // a text row touching the strip or the line below it
                reach = max(reach, up)
                found = true
                minX = min(minX, rowMin)
                maxX = max(maxX, rowMax)
            }
            up += rowStep
        }
        guard found else { return nil }
        let margin = 0.035                          // room for the soft edge so the whole line stays sharp-free
        return (CGFloat(min(0.34, Double(reach) / Double(h) + margin)),
                CGFloat(max(0.02, Double(minX) / Double(w) - 0.03)),
                CGFloat(min(0.98, Double(maxX + step) / Double(w) + 0.03)))
    }

    /// Auto mode: grow the strip at once when a caption needs more room, shrink it back ~1.5 s later.
    private func updateBand(for frame: CVPixelBuffer) {
        guard subtitleMode == .auto else { return }
        detectCounter += 1
        guard detectCounter % 2 == 0 else { return }          // every other game frame is plenty
        var target = (top: CGFloat(0.13), left: CGFloat(0.2), right: CGFloat(0.8))
        if let c = findCaptions(in: frame) {
            target = (max(target.top, c.top), min(target.left, c.left), max(target.right, c.right))
        }
        // Round to steps so the mask is reused instead of redrawn for tiny changes.
        target = ((target.top * 100).rounded(.up) / 100, (target.left * 50).rounded(.down) / 50,
                  (target.right * 50).rounded(.up) / 50)
        if target.top > band.top || target.left < band.left || target.right > band.right {
            band = (max(band.top, target.top), min(band.left, target.left), max(band.right, target.right))
            bandHold = 24                                    // ≈ 1.5 s of game frames (checked every other)
        } else if bandHold > 0 {
            bandHold -= 1
        } else {
            band = target
        }
    }

    /// White = protected (keep the original), black = enhance. Soft edges; one mask per strip size, reused.
    private func hudMask() -> CIImage? {
        let key = "\(Int((band.top * 100).rounded()))-\(Int((band.left * 50).rounded()))-\(Int((band.right * 50).rounded()))"
        if masks[key] == nil {
            if masks.count > 40 { masks.removeAll() }
            let attributes: [String: Any] = [
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var buffer: CVPixelBuffer?
            guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_OneComponent8,
                                      attributes as CFDictionary, &buffer) == kCVReturnSuccess, let buffer else { return nil }
            let extent = CGRect(x: 0, y: 0, width: width, height: height)
            var mask = CIImage(color: .black).cropped(to: extent)
            for rect in hudRects() {
                mask = CIImage(color: .white).cropped(to: rect).composited(over: mask)
            }
            mask = mask.clampedToExtent().applyingGaussianBlur(sigma: 10).cropped(to: extent)
            context.render(mask, to: buffer, bounds: extent, colorSpace: nil)
            masks[key] = buffer
        }
        guard let maskBuffer = masks[key] else { return nil }
        return CIImage(cvPixelBuffer: maskBuffer, options: [.colorSpace: NSNull()])
    }

    /// Runs on the processing queue for every new game frame.
    /// `hurry` = processing is behind, so skip the expensive clean-up for this frame.
    func enhance(_ frame: CVPixelBuffer, at time: CMTime, hurry: Bool) -> CVPixelBuffer {
        guard level != .off, CVPixelBufferGetWidth(frame) == width, CVPixelBufferGetHeight(frame) == height,
              CVPixelBufferGetPixelFormatType(frame) == format else { return frame }
        // Every buffer gets the same HD (BT.709) colour tags, so nothing shifts brightness between passes.
        Enhancer.tagHD(frame)
        var base = frame
        if denoiseActive && !hurry {
            let start = CACurrentMediaTime()
            if let cleaned = denoise(frame, at: time) {
                base = cleaned
                denoiseMs += (CACurrentMediaTime() - start) * 1000
                denoiseCount += 1
            }
        } else if hurry {
            previous = nil
        }

        let start = CACurrentMediaTime()
        guard let output = newBuffer(outputPool) else { lastError = "couldn't allocate frame"; return base }
        CVBufferPropagateAttachments(frame, output)
        Enhancer.tagHD(output)
        let noColourManagement: [CIImageOption: Any] = [.colorSpace: NSNull()]
        let original = CIImage(cvPixelBuffer: frame, options: noColourManagement)
        var image = CIImage(cvPixelBuffer: base, options: noColourManagement)
        image = image.applyingFilter("CISharpenLuminance", parameters: [
            kCIInputSharpnessKey: level.sharpness,
            kCIInputRadiusKey: 1.5,
        ])
        image = image.applyingFilter("CIVibrance", parameters: ["inputAmount": level.vibrance])
        image = image.applyingFilter("CIColorControls", parameters: [
            kCIInputContrastKey: level.contrast,
            kCIInputSaturationKey: 1.0,
            kCIInputBrightnessKey: 0.0,
        ])
        image = image.cropped(to: original.extent)
        if protectHUD { updateBand(for: frame) }
        if protectHUD, let mask = hudMask() {
            // Soft-edged blend so there are no visible boxes where protection starts.
            image = original.applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: image,
                kCIInputMaskImageKey: mask,
            ]).cropped(to: original.extent)
        }
        context.render(image, to: output, bounds: original.extent, colorSpace: nil)
        pictureMs += (CACurrentMediaTime() - start) * 1000
        pictureCount += 1
        return output
    }

    static func tagHD(_ buffer: CVPixelBuffer) {
        CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
    }

    func resetTimings() {
        denoiseMs = 0; denoiseCount = 0
        pictureMs = 0; pictureCount = 0
    }
}

/// How to read the brightness range of the capture card's picture.
enum ColourRange: Int {
    case auto = 0      // trust what the card says
    case limited = 1   // treat as 16–235 (fixes a washed-out, grey-black picture)
    case full = 2      // treat as 0–255 (fixes crushed blacks and clipped whites)

    var forcedFormat: OSType? {
        switch self {
        case .auto: return nil
        case .limited: return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        case .full: return kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        }
    }
    var title: String {
        switch self {
        case .auto: return "Auto"
        case .limited: return "Limited (16–235)"
        case .full: return "Full (0–255)"
        }
    }
}

/// Copies capture-card frames into our own buffers so the card's small buffer pool is never starved.
/// With a forced colour range it copies the bytes unchanged and only changes the label (no conversion).
final class BufferCopier {
    private var pool: CVPixelBufferPool?
    private var session: VTPixelTransferSession?
    private var width = 0, height = 0
    private var format: OSType = 0

    private func ensurePool(width w: Int, height h: Int, format f: OSType) -> CVPixelBufferPool? {
        if pool == nil || w != width || h != height || f != format {
            width = w; height = h; format = f
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferPixelFormatTypeKey as String: f,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ]
            pool = nil
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        }
        return pool
    }

    private var scaledPool: CVPixelBufferPool?
    private var scaledKey = (0, 0, OSType(0))

    /// The largest size that fits in maxWidth × maxHeight with the same shape (even numbers).
    static func fitted(width: Int, height: Int, maxWidth: Int, maxHeight: Int) -> (Int, Int) {
        guard width > maxWidth || height > maxHeight else { return (width, height) }
        let scale = min(Double(maxWidth) / Double(width), Double(maxHeight) / Double(height))
        let w = Int((Double(width) * scale).rounded(.down)) & ~1
        let h = Int((Double(height) * scale).rounded(.down)) & ~1
        return (max(w, 2), max(h, 2))
    }

    /// Copies a frame, shrinking it to fit maxWidth × maxHeight when it's bigger
    /// (Apple's real-time frame interpolation only accepts up to 1920×1080).
    func copy(_ source: CVPixelBuffer, range: ColourRange = .auto, maxWidth: Int, maxHeight: Int) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(source), h = CVPixelBufferGetHeight(source)
        let (fw, fh) = BufferCopier.fitted(width: w, height: h, maxWidth: maxWidth, maxHeight: maxHeight)
        guard fw != w || fh != h else { return copy(source, range: range) }
        let sourceFormat = CVPixelBufferGetPixelFormatType(source)
        let biplanar420: Set<OSType> = [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                         kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        let relabel = range.forcedFormat != nil && biplanar420.contains(sourceFormat) && range.forcedFormat! != sourceFormat
        // Relabelling must happen on the original bytes first; otherwise scale straight from the source.
        guard let input = relabel ? copy(source, range: range) : source else { return nil }
        let format = CVPixelBufferGetPixelFormatType(input)
        if scaledPool == nil || scaledKey != (fw, fh, format) {
            scaledKey = (fw, fh, format)
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: fw,
                kCVPixelBufferHeightKey as String: fh,
                kCVPixelBufferPixelFormatTypeKey as String: format,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ]
            scaledPool = nil
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &scaledPool)
        }
        if session == nil { VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) }
        var out: CVPixelBuffer?
        guard let scaledPool, let session,
              CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, scaledPool, &out) == kCVReturnSuccess, let scaled = out,
              VTPixelTransferSessionTransferImage(session, from: input, to: scaled) == noErr else { return nil }
        CVBufferPropagateAttachments(input, scaled)
        return scaled
    }

    func copy(_ source: CVPixelBuffer, range: ColourRange = .auto) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(source), h = CVPixelBufferGetHeight(source)
        let sourceFormat = CVPixelBufferGetPixelFormatType(source)
        let biplanar420: Set<OSType> = [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                         kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        let target = (range.forcedFormat != nil && biplanar420.contains(sourceFormat)) ? range.forcedFormat! : sourceFormat
        guard let pool = ensurePool(width: w, height: h, format: target) else { return nil }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out) == kCVReturnSuccess, let copy = out else { return nil }

        if target != sourceFormat {
            // Same bytes, different range label.
            guard BufferCopier.copyPlanes(from: source, to: copy) else { return nil }
        } else {
            if session == nil { VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &session) }
            guard let session, VTPixelTransferSessionTransferImage(session, from: source, to: copy) == noErr else { return nil }
        }
        CVBufferPropagateAttachments(source, copy)
        return copy
    }

    private static func copyPlanes(from source: CVPixelBuffer, to destination: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        let planes = CVPixelBufferGetPlaneCount(source)
        guard planes == CVPixelBufferGetPlaneCount(destination), planes > 0 else { return false }
        for plane in 0..<planes {
            guard let from = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                  let to = CVPixelBufferGetBaseAddressOfPlane(destination, plane) else { return false }
            let fromStride = CVPixelBufferGetBytesPerRowOfPlane(source, plane)
            let toStride = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
            let rows = min(CVPixelBufferGetHeightOfPlane(source, plane), CVPixelBufferGetHeightOfPlane(destination, plane))
            let bytes = min(fromStride, toStride)
            for row in 0..<rows {
                memcpy(to + row * toStride, from + row * fromStride, bytes)
            }
        }
        return true
    }
}

// MARK: - Recording and screenshots

/// Records exactly what Smooth60 shows (generated frames included) plus the game audio, as an H.264 MP4.
final class Recorder {
    private let queue = DispatchQueue(label: "smooth60.recorder", qos: .userInitiated)
    private let lock = NSLock()
    private var active = false

    // Touched only on `queue`.
    private var pendingURL: URL?
    private var expectedFPS = 60
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var audioInput: AVAssetWriterInput?
    private var sessionStart = CMTime.invalid
    private var lastVideoTime = CMTime.invalid
    private var latestAudioFormat: CMAudioFormatDescription?
    private var transfer: VTPixelTransferSession?
    private var pool: CVPixelBufferPool?
    private var poolSize = (0, 0)
    private(set) var lastError: String?

    var isRecording: Bool {
        lock.lock(); defer { lock.unlock() }
        return active
    }

    /// Default: Desktop › Smooth60 › Recordings / Screenshots.
    static func defaultFolder(_ kind: String) -> URL {
        FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Smooth60", isDirectory: true)
            .appendingPathComponent(kind, isDirectory: true)
    }

    /// The folder chosen in the Display tab, or the default. kind = "Recordings" or "Screenshots".
    static func saveFolder(_ kind: String) -> URL {
        let key = kind == "Screenshots" ? "screenshotsFolder" : "recordingsFolder"
        if let path = UserDefaults.standard.string(forKey: key), !path.isEmpty {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            // An unplugged drive or deleted folder falls back to the default instead of failing.
            if (try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)) != nil { return url }
        }
        return defaultFolder(kind)
    }

    static func newFileURL(folder: String, name: String, ext: String) -> URL {
        let base = saveFolder(folder == "Pictures" ? "Screenshots" : "Recordings")
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        return base.appendingPathComponent("\(name) \(stamp.string(from: Date())).\(ext)")
    }

    func start(fps: Int) -> URL {
        let url = Recorder.newFileURL(folder: "Movies", name: "Smooth60", ext: "mp4")
        lock.lock(); active = true; lock.unlock()
        queue.async {
            self.pendingURL = url
            self.expectedFPS = fps
            self.lastError = nil
        }
        return url
    }

    func stop(completion: @escaping @Sendable (URL?, String?) -> Void) {
        lock.lock(); active = false; lock.unlock()
        queue.async {
            self.pendingURL = nil
            guard let writer = self.writer, writer.status == .writing else {
                let message = self.lastError ?? "Nothing was recorded"
                self.reset()
                completion(nil, message)
                return
            }
            self.videoInput?.markAsFinished()
            self.audioInput?.markAsFinished()
            let url = writer.outputURL
            writer.finishWriting {
                let error = writer.status == .completed ? nil : (writer.error?.localizedDescription ?? "Recording failed")
                completion(error == nil ? url : nil, error)
            }
            self.reset()
        }
    }

    private func reset() {
        writer = nil; videoInput = nil; adaptor = nil; audioInput = nil
        sessionStart = .invalid; lastVideoTime = .invalid
    }

    /// Called for every frame Smooth60 shows, with its content time.
    func appendVideo(_ pixelBuffer: CVPixelBuffer, at time: CMTime) {
        guard isRecording else { return }
        queue.async { self.writeVideo(pixelBuffer, at: time) }
    }

    /// Called for every chunk of game audio (already on the same clock as the video).
    func appendAudio(_ sampleBuffer: CMSampleBuffer) {
        if let format = CMSampleBufferGetFormatDescription(sampleBuffer) {
            queue.async { self.latestAudioFormat = format }
        }
        guard isRecording else { return }
        queue.async {
            guard let input = self.audioInput, self.sessionStart.isValid, input.isReadyForMoreMediaData,
                  CMTimeCompare(CMSampleBufferGetPresentationTimeStamp(sampleBuffer), self.sessionStart) >= 0 else { return }
            input.append(sampleBuffer)
        }
    }

    private func writeVideo(_ source: CVPixelBuffer, at time: CMTime) {
        if writer == nil, let url = pendingURL {
            pendingURL = nil
            setUpWriter(url: url, width: CVPixelBufferGetWidth(source), height: CVPixelBufferGetHeight(source), start: time)
        }
        guard let writer, writer.status == .writing, let input = videoInput, let adaptor else { return }
        guard !lastVideoTime.isValid || CMTimeCompare(time, lastVideoTime) > 0 else { return }
        guard input.isReadyForMoreMediaData, let frame = normalized(source) else { return }
        if adaptor.append(frame, withPresentationTime: time) {
            lastVideoTime = time
        } else {
            lastError = writer.error?.localizedDescription ?? "Couldn't write a frame"
        }
    }

    /// Every frame into one format (8-bit 4:2:0, limited range) so the encoder sees a consistent stream.
    private func normalized(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(source), h = CVPixelBufferGetHeight(source)
        if CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange { return source }
        if pool == nil || poolSize != (w, h) {
            poolSize = (w, h)
            let attributes: [String: Any] = [
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            ]
            pool = nil
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
        }
        if transfer == nil { VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transfer) }
        guard let pool, let transfer else { return nil }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &out) == kCVReturnSuccess, let frame = out,
              VTPixelTransferSessionTransferImage(transfer, from: source, to: frame) == noErr else { return nil }
        return frame
    }

    private func setUpWriter(url: URL, width: Int, height: Int, start: CMTime) {
        do {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
            let fps = expectedFPS
            let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoColorPropertiesKey: [
                    AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                    AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                    AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
                ],
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: min(80_000_000, (fps >= 120 ? 40_000_000 : 25_000_000) * max(1, width * height / (1920 * 1080))),
                    AVVideoExpectedSourceFrameRateKey: fps,
                    AVVideoMaxKeyFrameIntervalKey: fps * 2,
                    AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                ],
            ])
            video.expectsMediaDataInRealTime = true
            guard writer.canAdd(video) else { lastError = "Can't record video in this format"; return }
            writer.add(video)
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: nil)

            var audio: AVAssetWriterInput?
            if let format = latestAudioFormat,
               let description = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee {
                let channels = max(1, min(Int(description.mChannelsPerFrame), 2))
                let input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
                    AVFormatIDKey: kAudioFormatMPEG4AAC,
                    AVSampleRateKey: description.mSampleRate,
                    AVNumberOfChannelsKey: channels,
                    AVEncoderBitRateKey: 192_000,
                ])
                input.expectsMediaDataInRealTime = true
                if writer.canAdd(input) { writer.add(input); audio = input }
            }

            guard writer.startWriting() else {
                lastError = writer.error?.localizedDescription ?? "Couldn't start recording"
                return
            }
            writer.startSession(atSourceTime: start)
            self.writer = writer
            self.videoInput = video
            self.adaptor = adaptor
            self.audioInput = audio
            self.sessionStart = start
            self.lastVideoTime = .invalid
        } catch {
            lastError = error.localizedDescription
        }
    }
}

/// Saves a PNG of a frame to Desktop/Smooth60/Screenshots.
enum Screenshotter {
    static func save(_ pixelBuffer: CVPixelBuffer) -> URL? {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let data = CIContext().pngRepresentation(of: image, format: .RGBA8, colorSpace: space, options: [:])
        else { return nil }
        let url = Recorder.newFileURL(folder: "Pictures", name: "Smooth60", ext: "png")
        do { try data.write(to: url) } catch { return nil }
        return url
    }
}

// MARK: - Display

final class Presenter {
    private let renderer: AVSampleBufferVideoRenderer
    private var formatDescription: CMVideoFormatDescription?
    let recorder = Recorder()

    // Extra screens showing the same frames (the stream output window).
    private let mirrorLock = NSLock()
    private var mirrors: [AVSampleBufferVideoRenderer] = []

    func addMirror(_ mirror: AVSampleBufferVideoRenderer) {
        mirrorLock.lock(); mirrors.append(mirror); mirrorLock.unlock()
    }
    func removeMirror(_ mirror: AVSampleBufferVideoRenderer) {
        mirrorLock.lock(); mirrors.removeAll { $0 === mirror }; mirrorLock.unlock()
    }

    // Before / After: the untouched frames go to a second layer.
    private let beforeLock = NSLock()
    private var beforeRenderer: AVSampleBufferVideoRenderer?
    private var beforeFormat: CMVideoFormatDescription?

    func setBefore(_ renderer: AVSampleBufferVideoRenderer?) {
        beforeLock.lock(); beforeRenderer = renderer; beforeFormat = nil; beforeLock.unlock()
    }

    /// Called on the capture thread with each new (untouched) game frame.
    func showBefore(_ pixelBuffer: CVPixelBuffer, at time: CMTime) {
        beforeLock.lock(); defer { beforeLock.unlock() }
        guard let renderer = beforeRenderer else { return }
        if beforeFormat == nil || !CMVideoFormatDescriptionMatchesImageBuffer(beforeFormat!, imageBuffer: pixelBuffer) {
            beforeFormat = nil
            CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer,
                                                         formatDescriptionOut: &beforeFormat)
        }
        guard let format = beforeFormat else { return }
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: time, decodeTimeStamp: .invalid)
        var sample: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer,
                                                 formatDescription: format, sampleTiming: &timing, sampleBufferOut: &sample)
        guard let sample else { return }
        if renderer.status == .failed { renderer.flush() }
        renderer.enqueue(sample)
    }

    // For screenshots (read from the main thread).
    private let lastLock = NSLock()
    private var lastFrame: CVPixelBuffer?

    // For the 1% low: gaps between shown frames over the last ~10 seconds (processing queue only).
    private var lastShownTime = CMTime.invalid
    private var frameGaps = [Double]()

    init(renderer: AVSampleBufferVideoRenderer) {
        self.renderer = renderer
    }

    var latestFrame: CVPixelBuffer? {
        lastLock.lock(); defer { lastLock.unlock() }
        return lastFrame
    }

    /// 1% low frame rate: the frame rate during the slowest 1% of moments (pauses over 0.25 s ignored).
    func onePercentLowFPS() -> Double? {
        guard frameGaps.count >= 100 else { return nil }
        let sorted = frameGaps.sorted()
        let gap = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.99))]
        return gap > 0 ? 1.0 / gap : nil
    }

    /// Queues a frame to appear when the playback clock reaches `time` (host-clock time of its content).
    func show(_ pixelBuffer: CVPixelBuffer, at time: CMTime) {
        if lastShownTime.isValid {
            let gap = CMTimeGetSeconds(CMTimeSubtract(time, lastShownTime))
            if gap > 0 && gap < 0.25 {
                frameGaps.append(gap)
                if frameGaps.count > 1200 { frameGaps.removeFirst(frameGaps.count - 1200) }
            }
        }
        lastShownTime = time
        lastLock.lock(); lastFrame = pixelBuffer; lastLock.unlock()
        recorder.appendVideo(pixelBuffer, at: time)

        if renderer.status == .failed { renderer.flush() }

        if formatDescription == nil ||
            !CMVideoFormatDescriptionMatchesImageBuffer(formatDescription!, imageBuffer: pixelBuffer) {
            formatDescription = nil
            CMVideoFormatDescriptionCreateForImageBuffer(allocator: kCFAllocatorDefault,
                                                         imageBuffer: pixelBuffer,
                                                         formatDescriptionOut: &formatDescription)
        }
        guard let format = formatDescription else { return }

        var timing = CMSampleTimingInfo(duration: .invalid,
                                        presentationTimeStamp: time,
                                        decodeTimeStamp: .invalid)
        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(allocator: kCFAllocatorDefault,
                                                 imageBuffer: pixelBuffer,
                                                 formatDescription: format,
                                                 sampleTiming: &timing,
                                                 sampleBufferOut: &sampleBuffer)
        guard let sampleBuffer else { return }
        renderer.enqueue(sampleBuffer)
        mirrorLock.lock(); let extra = mirrors; mirrorLock.unlock()
        for mirror in extra {
            if mirror.status == .failed { mirror.flush() }
            mirror.enqueue(sampleBuffer)
        }
    }
}

/// One clock for picture and sound. Everything plays `delay` after it was captured,
/// so generated frames get evenly spaced and the audio lines up with the video.
final class PlaybackClock {
    let synchronizer = AVSampleBufferRenderSynchronizer()
    /// Replaced with a fresh one if it fails (audioLock).
    private var audioRenderer = AVSampleBufferAudioRenderer()
    private let audioLock = NSLock()
    private var wantMuted = false, wantVolume: Float = 1, wantOutputUID: String?
    private var failures: [Date] = []
    private(set) var lastAudioError = ""
    private(set) var audioFormatText = "no sound yet"
    private(set) var delayMs: Double
    private let lock = NSLock()

    init(displayLayer: AVSampleBufferDisplayLayer, delayMs: Double) {
        self.delayMs = delayMs
        audioRenderer.audioTimePitchAlgorithm = .timeDomain   // tiny speed corrections without pitch change
        synchronizer.addRenderer(displayLayer)
        synchronizer.addRenderer(audioRenderer)
        anchor()
    }

    /// Mute / volume / output speaker, remembered so a replacement player gets them too.
    func setAudio(muted: Bool, volume: Float, outputUID: String?) {
        audioLock.lock(); defer { audioLock.unlock() }
        wantMuted = muted; wantVolume = volume; wantOutputUID = outputUID
        apply(to: audioRenderer)
    }

    var audioOutputUID: String? {
        audioLock.lock(); defer { audioLock.unlock() }
        return audioRenderer.audioOutputDeviceUniqueID
    }

    private func apply(to renderer: AVSampleBufferAudioRenderer) {
        renderer.isMuted = wantMuted
        renderer.volume = wantVolume
        if renderer.audioOutputDeviceUniqueID != wantOutputUID { renderer.audioOutputDeviceUniqueID = wantOutputUID }
    }

    /// A failed player is thrown away and replaced (flushing alone doesn't always bring it back).
    private func replaceRenderer() {
        synchronizer.removeRenderer(audioRenderer, at: synchronizer.currentTime(), completionHandler: nil)
        let fresh = AVSampleBufferAudioRenderer()
        fresh.audioTimePitchAlgorithm = .timeDomain
        apply(to: fresh)
        synchronizer.addRenderer(fresh)
        audioRenderer = fresh
    }

    private func delayTime() -> CMTime { CMTime(seconds: delayMs / 1000, preferredTimescale: 1_000_000) }

    /// Pins the clock to "host time minus delay".
    private func anchor() {
        let now = CMClockGetTime(CMClockGetHostTimeClock())
        synchronizer.setRate(1.0, time: CMTimeSubtract(now, delayTime()), atHostTime: now)
    }

    func setDelay(_ ms: Double) {
        lock.lock(); defer { lock.unlock() }
        delayMs = ms
        anchor()
    }

    /// The sound card's clock drifts a little from the system clock. Instead of jumping back
    /// (an audible cut-out), play a hair faster or slower until it's lined up again.
    func correctDriftIfNeeded() {
        lock.lock(); defer { lock.unlock() }
        let now = CMClockGetTime(CMClockGetHostTimeClock())
        let expected = CMTimeSubtract(now, delayTime())
        let actual = synchronizer.currentTime()
        let drift = CMTimeGetSeconds(CMTimeSubtract(actual, expected))     // + = running ahead
        if abs(drift) > 0.25 { anchor(); return }                            // way off (e.g. after sleep): jump
        let rate = abs(drift) < 0.002 ? 1.0 : min(1.01, max(0.99, 1.0 - drift / 2))   // close the gap in ~2 s
        if abs(Double(synchronizer.rate) - rate) > 0.0001 {
            synchronizer.setRate(Float(rate), time: actual, atHostTime: now)
        }
    }

    var audioStatus: String {
        audioLock.lock(); defer { audioLock.unlock() }
        switch audioRenderer.status {
        case .rendering: return "playing"
        case .failed: return "failed: \(lastAudioError)"
        default: return lastAudioError.isEmpty ? "waiting" : "recovering after: \(lastAudioError)"
        }
    }

    private static func describe(_ error: Error?) -> String {
        guard let error = error as NSError? else { return "unknown" }
        var text = "\(error.localizedDescription) [\(error.domain) \(error.code)]"
        if let under = error.userInfo[NSUnderlyingErrorKey] as? NSError { text += " ← \(under.domain) \(under.code)" }
        return text
    }

    func enqueueAudio(_ sampleBuffer: CMSampleBuffer) {
        audioLock.lock(); defer { audioLock.unlock() }
        if let format = CMSampleBufferGetFormatDescription(sampleBuffer),
           let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee {
            let isFloat = asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0
            let interleaved = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved == 0
            audioFormatText = String(format: "%.0f Hz · %d ch · %d-bit %@ · %@", asbd.mSampleRate, asbd.mChannelsPerFrame,
                                     asbd.mBitsPerChannel, isFloat ? "float" : "int", interleaved ? "interleaved" : "planar")
        }
        if audioRenderer.status == .failed {
            lastAudioError = PlaybackClock.describe(audioRenderer.error)
            failures.append(Date())
            failures.removeAll { Date().timeIntervalSince($0) > 10 }
            if failures.count >= 2 {
                replaceRenderer()          // keeps failing: start over with a new player
                failures.removeAll()
            } else {
                audioRenderer.flush()
            }
        }
        audioRenderer.enqueue(sampleBuffer)
    }
}

/// Moves every timestamp in a sample buffer by `offset`.
func retimed(_ sampleBuffer: CMSampleBuffer, by offset: CMTime) -> CMSampleBuffer? {
    var count: CMItemCount = 0
    CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: 0, arrayToFill: nil, entriesNeededOut: &count)
    guard count > 0 else { return nil }
    var timings = [CMSampleTimingInfo](repeating: CMSampleTimingInfo(), count: count)
    CMSampleBufferGetSampleTimingInfoArray(sampleBuffer, entryCount: count, arrayToFill: &timings, entriesNeededOut: &count)
    for i in timings.indices {
        timings[i].presentationTimeStamp = CMTimeAdd(timings[i].presentationTimeStamp, offset)
        if timings[i].decodeTimeStamp.isValid {
            timings[i].decodeTimeStamp = CMTimeAdd(timings[i].decodeTimeStamp, offset)
        }
    }
    var out: CMSampleBuffer?
    CMSampleBufferCreateCopyWithNewTiming(allocator: kCFAllocatorDefault, sampleBuffer: sampleBuffer,
                                          sampleTimingEntryCount: count, sampleTimingArray: &timings,
                                          sampleBufferOut: &out)
    return out
}

final class VideoView: NSView {
    let displayLayer = AVSampleBufferDisplayLayer()
    /// The untouched capture-card picture, shown left of the divider in Before / After mode.
    let beforeLayer = AVSampleBufferDisplayLayer()
    private let beforeMask = CALayer()
    private let divider = CALayer()
    private let handle = CALayer()
    /// Dashed outlines of the areas enhancement leaves alone (HUD corners + subtitle strip).
    private let protectedLayer = CAShapeLayer()
    var showsProtected = false { didSet { protectedLayer.isHidden = !showsProtected; needsLayout = true } }
    var protectedRects: [CGRect] = [] { didSet { if showsProtected { needsLayout = true } } }
    var contentSize = CGSize(width: 16, height: 9) { didSet { if showsProtected { needsLayout = true } } }
    let beforeTag = NSTextField(labelWithString: "BEFORE")
    let afterTag = NSTextField(labelWithString: "AFTER")
    /// 0…1 — where the divider sits. Drag it with the mouse.
    var split: CGFloat = 0.5 { didSet { needsLayout = true } }
    var comparing = false { didSet { applyComparing() } }

    private func applyComparing() {
        beforeLayer.isHidden = !comparing
        divider.isHidden = !comparing
        handle.isHidden = !comparing
        beforeTag.isHidden = !comparing
        afterTag.isHidden = !comparing
        window?.invalidateCursorRects(for: self)
        needsLayout = true
    }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        for l in [displayLayer, beforeLayer] {
            l.videoGravity = .resizeAspect
            l.backgroundColor = NSColor.black.cgColor
            l.frame = bounds
            layer?.addSublayer(l)
        }
        displayLayer.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
        beforeMask.backgroundColor = NSColor.white.cgColor
        beforeLayer.mask = beforeMask
        divider.backgroundColor = NSColor.white.cgColor
        divider.shadowOpacity = 0.6
        divider.shadowRadius = 3
        divider.shadowOffset = .zero
        handle.backgroundColor = NSColor(srgbRed: 0.91, green: 0.11, blue: 0.16, alpha: 1).cgColor
        handle.borderColor = NSColor.white.cgColor
        handle.borderWidth = 2
        handle.cornerRadius = 14
        protectedLayer.fillColor = NSColor(srgbRed: 0.91, green: 0.11, blue: 0.16, alpha: 0.10).cgColor
        protectedLayer.strokeColor = NSColor(srgbRed: 1, green: 0.25, blue: 0.3, alpha: 0.95).cgColor
        protectedLayer.lineWidth = 2
        protectedLayer.lineDashPattern = [6, 4]
        protectedLayer.isHidden = true
        layer?.addSublayer(protectedLayer)
        layer?.addSublayer(divider)
        layer?.addSublayer(handle)
        for (tag, colour) in [(beforeTag, NSColor(white: 0.08, alpha: 0.85)),
                              (afterTag, NSColor(srgbRed: 0.91, green: 0.11, blue: 0.16, alpha: 0.92))] {
            tag.font = NSFont.systemFont(ofSize: 12, weight: .bold)
            tag.textColor = .white
            tag.backgroundColor = colour
            tag.drawsBackground = true
            tag.alignment = .center
            addSubview(tag)
        }
        applyComparing()   // property observers don't run inside init
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    override func layout() {
        super.layout()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        displayLayer.frame = bounds
        beforeLayer.frame = bounds
        let x = (bounds.width * split).rounded()
        beforeMask.frame = CGRect(x: 0, y: 0, width: x, height: bounds.height)
        divider.frame = CGRect(x: x - 1, y: 0, width: 2, height: bounds.height)
        handle.frame = CGRect(x: x - 14, y: bounds.midY - 14, width: 28, height: 28)
        protectedLayer.frame = bounds
        if showsProtected {
            // Same aspect-fit rectangle the video is drawn in.
            let scale = min(bounds.width / contentSize.width, bounds.height / contentSize.height)
            let vw = contentSize.width * scale, vh = contentSize.height * scale
            let video = CGRect(x: (bounds.width - vw) / 2, y: (bounds.height - vh) / 2, width: vw, height: vh)
            let path = CGMutablePath()
            for r in protectedRects {
                path.addRect(CGRect(x: video.minX + r.minX * vw, y: video.minY + r.minY * vh,
                                    width: r.width * vw, height: r.height * vh).insetBy(dx: 1, dy: 1))
            }
            protectedLayer.path = path
        }
        CATransaction.commit()
        for tag in [beforeTag, afterTag] { tag.sizeToFit() }
        let pad: CGFloat = 10
        let h = beforeTag.frame.height + 6
        let top = bounds.height - h - pad      // tags sit at the top, either side of the divider
        beforeTag.frame = NSRect(x: max(pad, x - beforeTag.frame.width - 16 - pad), y: top,
                                 width: beforeTag.frame.width + 16, height: h)
        afterTag.frame = NSRect(x: min(bounds.width - afterTag.frame.width - 16 - pad, x + pad), y: top,
                                width: afterTag.frame.width + 16, height: h)
    }

    private func moveSplit(_ event: NSEvent) {
        guard comparing, bounds.width > 0 else { return }
        let p = convert(event.locationInWindow, from: nil)
        split = min(0.97, max(0.03, p.x / bounds.width))
    }
    override func mouseDown(with event: NSEvent) { comparing ? moveSplit(event) : super.mouseDown(with: event) }
    override func mouseDragged(with event: NSEvent) { comparing ? moveSplit(event) : super.mouseDragged(with: event) }
    override func resetCursorRects() {
        if comparing { addCursorRect(bounds, cursor: .resizeLeftRight) }
    }
}

// MARK: - Speakers / headphones (Core Audio output devices)

enum AudioOutputs {
    struct Device { let uid: String; let name: String }

    private static func address(_ selector: AudioObjectPropertySelector,
                                _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }

    private static func string(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var a = address(selector)
        var value: CFString? = nil
        var size = UInt32(MemoryLayout<CFString?>.size)
        let status = withUnsafeMutablePointer(to: &value) { AudioObjectGetPropertyData(id, &a, 0, nil, &size, $0) }
        guard status == noErr, let value else { return nil }
        return value as String
    }

    /// Every device that can play sound (built-in speakers, headphones, HDMI/DisplayPort monitors, USB, AirPods…).
    static func all() -> [Device] {
        var a = address(kAudioHardwarePropertyDevices)
        var size: UInt32 = 0
        let system = AudioObjectID(kAudioObjectSystemObject)
        guard AudioObjectGetPropertyDataSize(system, &a, 0, nil, &size) == noErr, size > 0 else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(system, &a, 0, nil, &size, &ids) == noErr else { return [] }
        return ids.compactMap { id in
            var streams = address(kAudioDevicePropertyStreams, kAudioDevicePropertyScopeOutput)
            var streamSize: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(id, &streams, 0, nil, &streamSize) == noErr, streamSize > 0,
                  let uid = string(id, kAudioDevicePropertyDeviceUID),
                  let name = string(id, kAudioObjectPropertyName) else { return nil }
            return Device(uid: uid, name: name)
        }
    }

    /// The Mac's current default output (System Settings → Sound → Output).
    static func defaultName() -> String? {
        var a = address(kAudioHardwarePropertyDefaultOutputDevice)
        var id = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &id) == noErr,
              id != 0 else { return nil }
        return string(id, kAudioObjectPropertyName)
    }
}

// MARK: - Capture + pipeline

/// One second of numbers for the panels at the bottom of the window.
struct LiveStats: Sendable {
    var device = ""
    var audio = ""
    var inputWidth = 0, inputHeight = 0
    var processWidth = 0, processHeight = 0          // smaller than input for 4K / 1440p capture cards
    var protectedRects: [CGRect] = []                // fractions of the frame, origin bottom-left
    var subtitleStrip = 0                            // % of frame height protected for subtitles
    var outputWidth = 0, outputHeight = 0
    var inputFPS = 0.0, gameFPS = 0.0, outputFPS = 0.0
    var onePercentLow: Double?
    var latencyMs = 0.0
    var screenHz = 60
    var frameGeneration = "Off"
    var upscaling = "Off"
    var enhancement = "Off"
    var budgetMs = 33.3
    var cleanUpMs = 0.0, pictureMs = 0.0
    var generateMs = 0.0
    var upscaleMs = 0.0, upscaledFramesPerGameFrame = 0
    var lateFrames = 0, skipped = 0
    var audioPacketsPerSecond = 0.0
    var audioPlayback = ""

    /// The two lanes run side by side; the busier one decides whether we keep up.
    var enhanceLaneMs: Double { cleanUpMs + pictureMs }
    var generateLaneMs: Double { generateMs + upscaleMs * Double(upscaledFramesPerGameFrame) }
    var load: Double { max(enhanceLaneMs, generateLaneMs) / max(budgetMs, 1) }
}

final class CaptureController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                               AVCaptureAudioDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "smooth60.capture", qos: .userInteractive)
    private let audioQueue = DispatchQueue(label: "smooth60.audio", qos: .userInteractive)
    private let processingQueue = DispatchQueue(label: "smooth60.processing", qos: .userInteractive)
    private let counterLock = NSLock()
    private var backlog = 0          // new frames handed to processingQueue but not finished yet
    private var skippedCount = 0
    private let presenter: Presenter
    private let clock: PlaybackClock
    private let detector = DuplicateDetector()

    // Pipeline state — touched only on `queue`.
    private var frameGenerationOn = true
    private var generator: FrameGenerator?
    private var generatorFailed: String?
    private var generatorFailures = 0          // failed interpolations in a row
    private var generatorRestarts = 0          // fresh sessions tried since the last success
    private var generateAt1080 = false         // big inputs (4K) that the engine refused: generate at 1080p
    private var previousSource: (buffer: CVPixelBuffer, time: CMTime)?
    private var tick = 0
    // Delay tuning: how long after capture each frame is actually ready to show.
    private var displayRefresh = 60
    private var colourRange = ColourRange.auto        // capture thread only
    // Upscaling (processing queue only)
    private var upscaleTarget = 0                     // output height, 0 = off
    // Enhancement runs on its own lane (enhanceQueue) so it overlaps with frame generation.
    private let enhanceQueue = DispatchQueue(label: "smooth60.enhance", qos: .userInteractive)
    private var enhanceBacklog = 0                    // counterLock
    private var enhanceLevel = EnhanceLevel.off       // enhanceQueue only
    private var protectHUD = true                     // enhanceQueue only
    private var subtitleMode = SubtitleMode.auto      // enhanceQueue only
    private var enhancer: Enhancer?                   // enhanceQueue only
    private var enhanceDenoiseFellBack = false        // enhanceQueue only
    private var enhanceSecondsBehind = 0              // processingQueue only
    private struct EnhanceStats {
        var level = EnhanceLevel.off
        var hud = true
        var denoiseMs = 0.0, denoiseCount = 0, pictureMs = 0.0, pictureCount = 0
        var hurried = 0
        var denoising = false
        var fellBack = false
        var denoiseFailed: String?
        var lastError: String?
        var protectedRects: [CGRect] = []
        var subtitleTop: CGFloat = 0
    }
    private var enhanceStats = EnhanceStats()         // counterLock
    private var upscaler: Upscaler?
    private var upscaleFailed: String?
    private var upscaleFellBack = false
    private var upscaleSecondsBehind = 0
    private var upscaleMsTotal = 0.0
    private var upscaleCount = 0
    private var targetFramesPerGap = 1      // 1 → 60 fps, 3 → 120 fps (chosen on the dashboard)
    private var fellBackTo60 = false        // 120 was chosen but this Mac couldn't keep up
    private var secondsBehind = 0
    private var unsupported120 = false
    private var neededMsThisSecond = 0.0
    private var neededHistory = [Double]()
    private var lateFrames = 0
    private var lastUniqueTick = 0
    private var averageUniqueInterval = 1.0 / 30.0
    private var lastUniqueTime = CMTime.invalid
    private let copier = BufferCopier()
    /// Biggest frame size frame generation can take (Apple's limit is 1920×1080; lowered automatically
    /// if a Mac rejects even that). Bigger capture cards (4K, 1440p) are scaled down to this first,
    /// and MetalFX upscaling can bring the result back up. counterLock.
    private var processingCap = (1920, 1080)
    private var processedSize = (0, 0)               // processingQueue
    private var attempts = 0
    private var slowestMs = 0.0

    // Stats
    private var inputCount = 0, uniqueCount = 0, outputCount = 0, generatedCount = 0
    private var processingMsTotal = 0.0
    private var statsStart = CACurrentMediaTime()
    private var resolutionText = ""
    private(set) var audioDeviceName: String?
    private var checkCounter = 0
    private var checkResults = [String]()
    private var lastCheckDetail = ""
    private var videoDeviceName = ""

    /// (simple stats, detailed stats, numbers for the panels)
    var onStats: (@MainActor (String, String, LiveStats) -> Void)?

    init(presenter: Presenter, clock: PlaybackClock) {
        self.presenter = presenter
        self.clock = clock
        super.init()
    }

    /// Capture timestamps → system (host) clock, which the playback clock follows.
    private func hostTime(_ time: CMTime) -> CMTime {
        guard let captureClock = session.synchronizationClock else { return time }
        return CMSyncConvertTime(time, from: captureClock, to: CMClockGetHostTimeClock())
    }

    /// Records how late (ms after capture) a frame became ready, for tuning the delay.
    private func noteReady(contentTime: CMTime) {
        let now = CMClockGetTime(CMClockGetHostTimeClock())
        let neededMs = CMTimeGetSeconds(CMTimeSubtract(now, contentTime)) * 1000
        guard neededMs > -50, neededMs < 500 else { return }
        neededMsThisSecond = max(neededMsThisSecond, neededMs)
        if neededMs > clock.delayMs { lateFrames += 1 }
    }

    /// Capture cards first; virtual cameras (OBS etc.) last.
    static func videoDevices() -> [AVCaptureDevice] {
        let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.external],
                                                       mediaType: .video,
                                                       position: .unspecified).devices
        return devices.enumerated()
            .sorted { (deviceRank($0.element), $0.offset) < (deviceRank($1.element), $1.offset) }
            .map { $0.element }
    }

    /// 0 = likely a capture card, 1 = webcam, 2 = iPhone/iPad (Continuity Camera), 3 = virtual camera.
    static func deviceRank(_ d: AVCaptureDevice) -> Int {
        let n = (d.localizedName + " " + d.modelID).lowercased()
        if n.contains("virtual") || n.contains("obs") || n.contains("camera extension") { return 3 }
        if d.isContinuityCamera || n.contains("iphone") || n.contains("ipad") || n.contains("desk view") { return 2 }
        if n.contains("facetime") || n.contains("webcam") { return 1 }
        return 0
    }

    /// "Kreo Lynx Video" → "kreo lynx", so it matches "Kreo Lynx Audio".
    static func nameStem(_ name: String) -> String {
        let drop: Set<String> = ["video", "audio", "capture", "input", "output", "(video)", "(audio)", "-"]
        return name.lowercased()
            .split(separator: " ")
            .map(String.init)
            .filter { !drop.contains($0) }
            .joined(separator: " ")
    }

    static func matchingAudioDevice(for video: AVCaptureDevice) -> AVCaptureDevice? {
        let audio = audioDevices()
        let stem = nameStem(video.localizedName)
        if !stem.isEmpty, let match = audio.first(where: { nameStem($0.localizedName) == stem }) { return match }
        let videoName = video.localizedName.lowercased()
        if let match = audio.first(where: {
            let n = $0.localizedName.lowercased()
            return n.contains(videoName) || videoName.contains(n) || (!stem.isEmpty && n.contains(stem))
        }) { return match }
        // Same USB vendor/product ID in the model string (UVC devices).
        if let vid = video.modelID.range(of: "VendorID_[0-9]+", options: .regularExpression).map({ String(video.modelID[$0]) }),
           let match = audio.first(where: { $0.modelID.contains(vid) }) { return match }
        return nil
    }

    static func audioDevices() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(deviceTypes: [.microphone, .external],
                                         mediaType: .audio,
                                         position: .unspecified).devices
    }

    /// Prefer 1080p at 60 fps, uncompressed (not MJPEG).
    static func bestFormat(for device: AVCaptureDevice) -> (AVCaptureDevice.Format, AVFrameRateRange)? {
        var best: (AVCaptureDevice.Format, AVFrameRateRange, Int)?
        for format in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard let range = format.videoSupportedFrameRateRanges.max(by: { $0.maxFrameRate < $1.maxFrameRate }) else { continue }
            let subtype = CMFormatDescriptionGetMediaSubType(format.formatDescription)
            let isMJPEG = subtype == kCMVideoCodecType_JPEG || subtype == kCMVideoCodecType_JPEG_OpenDML
            var score = 0
            if range.maxFrameRate >= 59 { score += 10_000 }
            if dims.width == 1920 && dims.height == 1080 { score += 5_000 }
            else if dims.height <= 1080 { score += Int(dims.height) }
            if !isMJPEG { score += 500 }
            if best == nil || score > best!.2 { best = (format, range, score) }
        }
        if let best { return (best.0, best.1) }
        return nil
    }

    /// The refresh rate of the screen the window is on; 110 Hz+ turns on 120 fps output.
    /// 0 = off, otherwise the output height (1440 or 2160). Used at 60 fps only.
    func setUpscale(targetHeight: Int) {
        processingQueue.async {
            self.upscaleTarget = targetHeight
            self.upscaler = nil
            self.upscaleFailed = nil
            self.upscaleFellBack = false
            self.upscaleSecondsBehind = 0
        }
    }

    func setEnhancement(_ level: EnhanceLevel, protectHUD: Bool, subtitles: SubtitleMode = .auto) {
        enhanceQueue.async {
            self.enhanceLevel = level
            self.protectHUD = protectHUD
            self.subtitleMode = subtitles
            self.enhancer = nil
            self.enhanceDenoiseFellBack = false
            self.counterLock.lock()
            self.enhanceStats = EnhanceStats()
            self.enhanceStats.level = level
            self.enhanceStats.hud = protectHUD
            self.counterLock.unlock()
        }
        processingQueue.async { self.enhanceSecondsBehind = 0 }
    }

    private func turnOffCleanUp() {
        enhanceQueue.async {
            self.enhanceDenoiseFellBack = true
            self.enhancer?.denoiseAllowed = false
            self.counterLock.lock(); self.enhanceStats.fellBack = true; self.counterLock.unlock()
        }
    }

    private func enhanced(_ frame: CVPixelBuffer, at time: CMTime, hurry: Bool) -> CVPixelBuffer {
        guard enhanceLevel != .off else { return frame }
        let w = CVPixelBufferGetWidth(frame), h = CVPixelBufferGetHeight(frame)
        let format = CVPixelBufferGetPixelFormatType(frame)
        if enhancer == nil || enhancer!.width != w || enhancer!.height != h || enhancer!.format != format {
            enhancer = Enhancer(width: w, height: h, format: format, level: enhanceLevel, protectHUD: protectHUD,
                                subtitleMode: subtitleMode)
            enhancer?.denoiseAllowed = !enhanceDenoiseFellBack
        }
        guard let enhancer else { return frame }
        let wantsCleanUp = enhancer.denoiseActive
        let result = enhancer.enhance(frame, at: time, hurry: hurry)
        counterLock.lock()
        enhanceStats.denoiseMs += enhancer.denoiseMs
        enhanceStats.denoiseCount += enhancer.denoiseCount
        enhanceStats.pictureMs += enhancer.pictureMs
        enhanceStats.pictureCount += enhancer.pictureCount
        if hurry && wantsCleanUp { enhanceStats.hurried += 1 }
        enhanceStats.denoising = enhancer.denoiseActive
        enhanceStats.denoiseFailed = enhancer.denoiseFailed
        enhanceStats.lastError = enhancer.lastError
        enhanceStats.protectedRects = protectHUD ? enhancer.protectedRects : []
        enhanceStats.subtitleTop = protectHUD ? enhancer.band.top : 0
        counterLock.unlock()
        enhancer.resetTimings()
        return result
    }

    private var upscalingActive: Bool {
        upscaleTarget > 0 && !upscaleFellBack && upscaleFailed == nil && (targetFramesPerGap == 1 || fellBackTo60)
    }

    private func ensureUpscaler(for frame: CVPixelBuffer) -> Upscaler? {
        let w = CVPixelBufferGetWidth(frame), h = CVPixelBufferGetHeight(frame)
        guard upscaleTarget > h else { return nil }   // never "upscale" to a smaller size
        if let upscaler, upscaler.inputWidth == w, upscaler.inputHeight == h, upscaler.targetHeight == upscaleTarget { return upscaler }
        do {
            upscaler = try Upscaler(width: w, height: h, targetHeight: upscaleTarget)
        } catch {
            upscaler = nil
            upscaleFailed = "\(error)"
        }
        return upscaler
    }

    /// Shows a frame, upscaled first when upscaling is on.
    private func deliver(_ frame: CVPixelBuffer, at time: CMTime) {
        var output = frame
        if upscalingActive, let upscaler = ensureUpscaler(for: frame) {
            let start = CACurrentMediaTime()
            if let big = upscaler.upscale(frame, at: time) {
                output = big
                upscaleCount += 1
                upscaleMsTotal += (CACurrentMediaTime() - start) * 1000
            }
        }
        presenter.show(output, at: time)
    }

    func setColourRange(_ range: ColourRange) {
        queue.async { self.colourRange = range }
    }

    var presenterForUI: Presenter { presenter }

    /// The refresh rate of the screen the video window is on (for warnings only).
    func setDisplayRefresh(_ hz: Int) {
        processingQueue.async { self.displayRefresh = hz }
    }

    /// 60 or 120, as chosen on the dashboard.
    func setTargetFPS(_ fps: Int) {
        processingQueue.async {
            self.targetFramesPerGap = fps >= 120 ? 3 : 1
            self.fellBackTo60 = false
            self.unsupported120 = false
            self.secondsBehind = 0
            self.previousSource = nil
        }
    }

    var onNotice: (@MainActor (String) -> Void)?

    func setFrameGeneration(_ on: Bool) {
        processingQueue.async {
            self.frameGenerationOn = on
            self.previousSource = nil
        }
    }

    func start(device: AVCaptureDevice) throws {
        session.stopRunning()
        videoDeviceName = device.localizedName
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }

        let videoInput = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(videoInput) { session.addInput(videoInput) }

        // Ask for the card's own range (full or limited) so nothing gets squashed before the colour-range setting.
        let videoOutput = AVCaptureVideoDataOutput()
        var pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        if let best = CaptureController.bestFormat(for: device),
           CMFormatDescriptionGetMediaSubType(best.0.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
            pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        }
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        // Game audio: use the capture card's own audio device (same name), never the Mac's mic.
        audioDeviceName = nil
        if let audioDevice = CaptureController.matchingAudioDevice(for: device),
           let audioInput = try? AVCaptureDeviceInput(device: audioDevice), session.canAddInput(audioInput) {
            session.addInput(audioInput)
            // Audio goes through the shared playback clock so it's delayed exactly as much as the video.
            let audioOutput = AVCaptureAudioDataOutput()
            // Ask for plain 48 kHz stereo float — the format the sound player always accepts,
            // whatever the capture card sends (some send 16-bit, 24-bit, 8 channels…).
            audioOutput.audioSettings = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 48_000,
                AVNumberOfChannelsKey: 2,
                AVLinearPCMBitDepthKey: 32,
                AVLinearPCMIsFloatKey: true,
                AVLinearPCMIsNonInterleaved: false,
                AVLinearPCMIsBigEndianKey: false,
            ]
            audioOutput.setSampleBufferDelegate(self, queue: audioQueue)
            if session.canAddOutput(audioOutput) { session.addOutput(audioOutput) }
            audioDeviceName = audioDevice.localizedName
        }
        session.commitConfiguration()

        applyBestFormat(device)
        // Fresh start: allow up to 1080p again (a hidden "maxProcessHeight" setting can lower it for testing).
        let testHeight = UserDefaults.standard.integer(forKey: "maxProcessHeight")
        counterLock.lock()
        processingCap = testHeight >= 360 && testHeight < 1080 ? (testHeight * 16 / 9, testHeight) : (1920, 1080)
        counterLock.unlock()
        processingQueue.async {
            self.generator = nil
            self.generatorFailed = nil
            self.generatorFailures = 0
            self.generatorRestarts = 0
            self.generateAt1080 = false
            self.previousSource = nil
        }
        session.startRunning()
        applyBestFormat(device)
    }

    private func applyBestFormat(_ device: AVCaptureDevice) {
        guard let best = CaptureController.bestFormat(for: device) else { return }
        let format = best.0, range = best.1
        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            device.activeVideoMinFrameDuration = range.minFrameDuration
            device.activeVideoMaxFrameDuration = range.minFrameDuration
            device.unlockForConfiguration()
        } catch {
            NSLog("Smooth60: couldn't set format: \(error)")
        }
    }

    func stop() { session.stopRunning() }

    // Video: called ~60 times a second on `queue` (the capture thread). Audio: on `audioQueue`.
    // This must stay fast, so the capture card is never held up: it only drops repeated frames,
    // copies new ones and hands them to `processingQueue`, where frame generation happens.
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output is AVCaptureAudioDataOutput {
            handleAudio(sampleBuffer)
            return
        }
        guard let raw = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let time = hostTime(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        let w = CVPixelBufferGetWidth(raw), h = CVPixelBufferGetHeight(raw)
        counterLock.lock(); inputCount += 1; counterLock.unlock()

        // Repeated frame (30 fps game in a 60 Hz signal): nothing new to show.
        if detector.isDuplicate(raw) {
            processingQueue.async { self.publishStatsIfNeeded(width: w, height: h) }
            return
        }
        counterLock.lock(); let cap = processingCap; counterLock.unlock()
        guard let copy = copier.copy(raw, range: colourRange, maxWidth: cap.0, maxHeight: cap.1) else { return }
        presenter.showBefore(copy, at: time)   // Before / After view (no-op when it's off)
        // Two lanes: enhancement on enhanceQueue, then frame generation + upscaling on processingQueue.
        // While frame N is being generated, frame N+1 is already being enhanced.
        counterLock.lock(); enhanceBacklog += 1; counterLock.unlock()
        enhanceQueue.async {
            self.counterLock.lock(); let hurry = self.enhanceBacklog > 1; self.counterLock.unlock()
            let frame = self.enhanced(copy, at: time, hurry: hurry)
            self.counterLock.lock(); self.enhanceBacklog -= 1; self.backlog += 1; self.counterLock.unlock()
            self.processingQueue.async {
                self.process(frame, time: time)
                self.counterLock.lock(); self.backlog -= 1; self.counterLock.unlock()
                self.publishStatsIfNeeded(width: w, height: h)
            }
        }
    }

    /// Runs on `processingQueue` for every new game frame (~30 a second at 30 fps).
    private func process(_ frame: CVPixelBuffer, time: CMTime) {
        uniqueCount += 1
        processedSize = (CVPixelBufferGetWidth(frame), CVPixelBufferGetHeight(frame))
        // This Mac refuses frame generation at this size: step down to 720p and try again.
        if let generator, generator.sizeRejections >= 5 {
            counterLock.lock()
            let smaller = processingCap.1 > 720 ? (1280, 720) : processingCap
            let changed = smaller != processingCap
            processingCap = smaller
            counterLock.unlock()
            self.generator = nil
            previousSource = nil
            if changed, let notice = onNotice {
                Task { @MainActor in notice("This Mac can't generate frames at this size — processing at 1280×720") }
            }
            if !changed { generatorFailed = "this Mac's frame interpolation rejected every size" }
        }
        // Measure the game's frame rate from timestamps.
        if lastUniqueTime.isValid {
            let seconds = CMTimeGetSeconds(CMTimeSubtract(time, lastUniqueTime))
            if seconds > 0 && seconds < 0.2 {
                averageUniqueInterval = averageUniqueInterval * 0.85 + seconds * 0.15
            }
        }
        lastUniqueTime = time

        // If a newer frame is already waiting, we've fallen behind: show this one without generating.
        counterLock.lock(); let behind = backlog > 1; counterLock.unlock()

        // Only generate frames when the game is running below ~40 fps.
        let gameIsSlow = averageUniqueInterval > 0.025
        guard frameGenerationOn, gameIsSlow, let generator = ensureGenerator(for: frame),
              let source = generator.makeSourceCopy(of: frame) else {
            deliver(frame, at: time)
            noteReady(contentTime: time)
            outputCount += 1
            previousSource = nil
            return
        }

        // Each frame is stamped with when its content happened; the playback clock shows it
        // a fixed delay later. Generated frames go evenly spaced between the two real ones.
        if behind {
            skippedCount += 1
        } else if let previous = previousSource,
                  CMTimeGetSeconds(CMTimeSubtract(time, previous.time)) < 0.1 {
            let start = CACurrentMediaTime()
            let generated = generator.interpolate(previous: previous.buffer, previousTime: previous.time,
                                                  current: source, currentTime: time)
            let ms = (CACurrentMediaTime() - start) * 1000
            processingMsTotal += ms
            attempts += 1
            slowestMs = max(slowestMs, ms)
            if generated == nil || generated?.isEmpty == true {
                generatorFailures += 1
                if generatorFailures >= 6 {
                    handleGeneratorFailure(inputWidth: CVPixelBufferGetWidth(frame), inputHeight: CVPixelBufferGetHeight(frame))
                }
            } else {
                generatorFailures = 0
                generatorRestarts = 0
            }
            if let generated, !generated.isEmpty {
                for (i, frame) in generated.enumerated() {
                    let t = FrameGenerator.time(i, of: generated.count, from: previous.time, to: time)
                    deliver(frame, at: t)
                    if i == 0 { noteReady(contentTime: t) }   // the earliest one has the tightest deadline
                }
                outputCount += generated.count
                generatedCount += generated.count
                checkCounter += 1
                if checkCounter % 30 == 0 { checkGenerated(previous.buffer, generated[generated.count / 2], source) }
            }
        }
        deliver(source, at: time)
        noteReady(contentTime: time)
        outputCount += 1
        previousSource = (source, time)
    }

    private var audioSyncMs = 0.0          // audioQueue only: user fine-tune (+ = later)
    private var audioPackets = 0           // counterLock

    /// Fine-tunes the sound against the picture: positive plays it later, negative earlier.
    func setAudioSync(ms: Double) { audioQueue.async { self.audioSyncMs = ms } }

    private func handleAudio(_ sampleBuffer: CMSampleBuffer) {
        counterLock.lock(); audioPackets += 1; counterLock.unlock()
        let original = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let offset = CMTimeAdd(CMTimeSubtract(hostTime(original), original),
                               CMTime(seconds: audioSyncMs / 1000, preferredTimescale: 1_000_000))
        if abs(CMTimeGetSeconds(offset)) < 0.0005 {
            clock.enqueueAudio(sampleBuffer)
            presenter.recorder.appendAudio(sampleBuffer)
        } else if let moved = retimed(sampleBuffer, by: offset) {
            clock.enqueueAudio(moved)
            presenter.recorder.appendAudio(moved)
        }
    }

    /// Once a second: keep the delay just above what the slowest frames need.
    private func tuneDelay() {
        guard neededMsThisSecond > 0 else {              // static screen: nothing new to learn
            clock.correctDriftIfNeeded()
            return
        }
        // Every delay change makes the sound jump (a tiny cut-out), so change it rarely:
        // raise with headroom when frames run late, tighten only after 20 s of being well ahead.
        neededHistory.append(neededMsThisSecond)
        if neededHistory.count > 20 { neededHistory.removeFirst() }
        neededMsThisSecond = 0
        let recentWorst = neededHistory.suffix(3).max() ?? 0
        let longWorst = neededHistory.max() ?? 0
        let current = clock.delayMs
        if recentWorst + 4 > current {
            clock.setDelay(min(140, (recentWorst + 16).rounded()))      // late: raise once, with room to spare
        } else if neededHistory.count == 20 && longWorst + 30 < current {
            clock.setDelay(max(20, (longWorst + 16).rounded()))         // well ahead for 20 s: tighten
            neededHistory.removeAll()
        } else {
            clock.correctDriftIfNeeded()
        }
    }

    /// Is the generated frame really "in between" the two real frames?
    private func checkGenerated(_ previous: CVPixelBuffer, _ middle: CVPixelBuffer, _ current: CVPixelBuffer) {
        let p = lumaSamples(previous), m = lumaSamples(middle), c = lumaSamples(current)
        let pc = meanDifference(p, c), pm = meanDifference(p, m), mc = meanDifference(m, c)
        guard pc >= 3.0 else { return }  // scene barely moving — differences are just noise
        let verdict: String
        if pm < 0.15 * pc { verdict = "SAME AS PREVIOUS" }
        else if mc < 0.15 * pc { verdict = "SAME AS CURRENT" }
        else if pm + mc > 2.0 * pc + 3.0 { verdict = "UNRELATED" }
        else { verdict = "OK" }
        checkResults.append(verdict)
        if checkResults.count > 10 { checkResults.removeFirst() }
        lastCheckDetail = String(format: "prev→mid %.1f · mid→cur %.1f · prev→cur %.1f", pm, mc, pc)
    }

    /// The size frames are generated at: the input size, or 1080p if the engine refused a bigger input.
    private func generationSize(_ w: Int, _ h: Int) -> (Int, Int) {
        guard generateAt1080, w > 1920 || h > 1080 else { return (w, h) }
        let fit = min(1920.0 / Double(w), 1080.0 / Double(h))
        let sw = Int((Double(w) * fit).rounded()) & ~1, sh = Int((Double(h) * fit).rounded()) & ~1
        return (sw, sh)
    }

    /// Interpolation keeps failing (e.g. "Processor is not initialized"): start a fresh session,
    /// then try 1080p for big inputs, and only then give up — the video keeps playing either way.
    private func handleGeneratorFailure(inputWidth w: Int, inputHeight h: Int) {
        let reason = generator?.lastError ?? "unknown error"
        generatorFailures = 0
        generator = nil
        previousSource = nil
        generatorRestarts += 1
        if generatorRestarts <= 2 { return }                       // just restart the engine
        if !generateAt1080 && (w > 1920 || h > 1080) {
            generateAt1080 = true
            generatorRestarts = 0
            if let notice = onNotice {
                Task { @MainActor in notice("Frame generation couldn't run at \(w)×\(h) — now generating at 1080p") }
            }
            return
        }
        generatorFailed = "Apple's frame interpolation won't start on this Mac (\(reason)). Video keeps playing at the game's own frame rate."
        if let notice = onNotice {
            Task { @MainActor in notice("Frame generation isn't working on this Mac — showing the game at its own frame rate") }
        }
    }

    private func ensureGenerator(for buffer: CVPixelBuffer) -> FrameGenerator? {
        let (w, h) = generationSize(CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
        let wanted = (targetFramesPerGap == 3 && !fellBackTo60) ? 3 : 1
        if let generator, generator.width == w, generator.height == h, generator.framesPerGap == wanted { return generator }
        if let generator, generator.width == w, generator.height == h, wanted == 3, unsupported120 { return generator }
        if generatorFailed != nil { return nil }
        generator = nil
        if wanted == 3 {
            if let g = try? FrameGenerator(width: w, height: h, framesPerGap: 3) {
                generator = g
                previousSource = nil
                return g
            }
            unsupported120 = true   // fall back to 60 fps below
        }
        do {
            generator = try FrameGenerator(width: w, height: h, framesPerGap: 1)
            previousSource = nil
        } catch {
            generator = nil
            // A big input (4K) the engine won't take: try again at 1080p before giving up.
            let iw = CVPixelBufferGetWidth(buffer), ih = CVPixelBufferGetHeight(buffer)
            if !generateAt1080 && (iw > 1920 || ih > 1080) {
                generateAt1080 = true
                if let notice = onNotice {
                    Task { @MainActor in notice("Frame generation couldn't run at \(iw)×\(ih) — now generating at 1080p") }
                }
                return ensureGenerator(for: buffer)
            }
            generatorFailed = "\(error)"
        }
        return generator
    }

    private func publishStatsIfNeeded(width w: Int, height h: Int) {
        let now = CACurrentMediaTime()
        let elapsed = now - statsStart
        guard elapsed >= 1.0 else { return }
        counterLock.lock(); let inputs = inputCount; inputCount = 0
        let packets = audioPackets; audioPackets = 0; counterLock.unlock()
        let inFps = Double(inputs) / elapsed
        let gameFps = Double(uniqueCount) / elapsed
        let outFps = Double(outputCount) / elapsed
        let avgMs = attempts > 0 ? processingMsTotal / Double(attempts) : 0

        var mode: String
        if !frameGenerationOn { mode = "Frame generation: OFF (⌘G to turn on)" }
        else if let failed = generatorFailed { mode = "Frame generation unavailable: \(failed)" }
        else if attempts > 0 {
            let perGap = generator?.framesPerGap ?? 1
            let budget = 33.3   // one game frame at 30 fps
            mode = String(format: "Frame generation: ON (%@) · %d made · avg %.1f ms, slowest %.1f ms (budget %.1f)",
                          perGap == 3 ? "30→120" : "30→60", generatedCount, avgMs, slowestMs, budget)
        }
        else { mode = String(format: "Frame generation: standby (game ≈ %.0f fps)", 1.0 / max(averageUniqueInterval, 0.001)) }
        if let err = generator?.lastError { mode += "\nLast error: \(err)" }
        // 120 fps chosen but falling behind for 3 seconds in a row: drop to 60 so it stays smooth.
        if targetFramesPerGap == 3 && !fellBackTo60 && generator?.framesPerGap == 3 {
            if skippedCount > max(2, uniqueCount / 10) { secondsBehind += 1 } else { secondsBehind = 0 }
            if secondsBehind >= 3 {
                fellBackTo60 = true
                previousSource = nil
                if let notice = onNotice {
                    Task { @MainActor in notice("This Mac can't keep up with 120 fps — switched to 60 fps") }
                }
            }
        }
        // Enhancement clean-up falling behind for 3 seconds: turn the clean-up off first (sharpen + colour stay).
        counterLock.lock()
        let es = enhanceStats
        enhanceStats.denoiseMs = 0; enhanceStats.denoiseCount = 0
        enhanceStats.pictureMs = 0; enhanceStats.pictureCount = 0
        enhanceStats.hurried = 0
        counterLock.unlock()
        let denoising = es.denoising
        if denoising {
            let threshold = max(2, uniqueCount / 10)
            if skippedCount > threshold || es.hurried > threshold { enhanceSecondsBehind += 1 } else { enhanceSecondsBehind = 0 }
            if enhanceSecondsBehind >= 3 {
                turnOffCleanUp()
                enhanceSecondsBehind = 0
                if let notice = onNotice {
                    Task { @MainActor in notice("This Mac can't keep up with clean-up — turned it off (sharpen + colour still on)") }
                }
            }
        }
        // Upscaling falling behind for 3 seconds in a row: turn it off so frame generation stays smooth.
        if upscalingActive && upscaler != nil && !denoising {
            if skippedCount > max(2, uniqueCount / 10) { upscaleSecondsBehind += 1 } else { upscaleSecondsBehind = 0 }
            if upscaleSecondsBehind >= 3 {
                upscaleFellBack = true
                if let notice = onNotice {
                    Task { @MainActor in notice("This Mac can't keep up with upscaling — turned it off") }
                }
            }
        }
        if upscaleTarget > 0 {
            if let failed = upscaleFailed {
                mode += "\nUpscaling unavailable: \(failed)"
            } else if upscaleFellBack {
                mode += "\nUpscaling turned off: this Mac couldn't keep up"
            } else if targetFramesPerGap == 3 && !fellBackTo60 {
                mode += "\nUpscaling is off at 120 fps (60 fps only)"
            } else if let upscaler, upscaleCount > 0 {
                mode += String(format: "\nUpscaling (MetalFX): %d×%d → %d×%d · %.1f ms per frame",
                               upscaler.inputWidth, upscaler.inputHeight, upscaler.outputWidth, upscaler.outputHeight,
                               upscaleMsTotal / Double(upscaleCount))
            }
            if let err = upscaler?.lastError { mode += " · \(err)" }
        }
        let upscaleAvgMs = upscaleCount > 0 ? upscaleMsTotal / Double(upscaleCount) : 0
        upscaleMsTotal = 0
        upscaleCount = 0
        if es.level != .off {
            var line = "\nEnhancement: \(es.level.name)"
            if es.denoiseCount > 0 {
                line += String(format: " · clean-up %.1f ms", es.denoiseMs / Double(es.denoiseCount))
            } else if let failed = es.denoiseFailed, es.level.denoiseStrength > 0 {
                line += " · clean-up unavailable (\(failed))"
            } else if es.fellBack && es.level.denoiseStrength > 0 {
                line += " · clean-up off (too slow)"
            }
            if es.pictureCount > 0 {
                line += String(format: " · sharpen + colour %.1f ms", es.pictureMs / Double(es.pictureCount))
            }
            if es.hud { line += " · HUD protected" }
            if let err = es.lastError { line += " · \(err)" }
            mode += line
        }
        if targetFramesPerGap == 3 && unsupported120 { mode += "\n120 fps isn't supported on this Mac — using 60" }
        if targetFramesPerGap == 3 && fellBackTo60 { mode += "\n120 fps was too heavy for this Mac — using 60" }
        if targetFramesPerGap == 3 && !fellBackTo60 && displayRefresh < 110 {
            mode += "\nThis screen runs at \(displayRefresh) Hz, so it can only show 60 of the 120 fps"
        } else if targetFramesPerGap == 3 && displayRefresh > 125 {
            mode += "\nTip: set this display to 120 Hz (System Settings → Displays) for perfectly even 120 fps"
        }
        if !checkResults.isEmpty {
            let ok = checkResults.filter { $0 == "OK" }.count
            mode += "\nCheck: \(ok)/\(checkResults.count) OK (last: \(checkResults.last!)) · \(lastCheckDetail)"
        }

        let late = lateFrames
        lateFrames = 0
        tuneDelay()
        mode += String(format: "\nPicture + sound delay: %.0f ms (in sync) · late frames: %d · skipped (behind): %d",
                       clock.delayMs, late, skippedCount)
        mode += String(format: "\nSound: %.0f packets/s from %@ (%@) · output %@ · %@",
                       Double(packets) / elapsed, audioDeviceName ?? "no audio device", clock.audioFormatText,
                       clock.audioOutputUID ?? "system default", clock.audioStatus)

        let source = "\(videoDeviceName) · Audio: \(audioDeviceName ?? "none found") · Screen: \(displayRefresh) Hz\n"
        let low = presenter.onePercentLowFPS().map { String(format: " · 1%% low %.0f fps", $0) } ?? ""
        let text = source + String(format: "%d×%d · Input %.0f fps · Game %.0f fps · Output %.0f fps", w, h, inFps, gameFps, outFps)
            + low + "\n" + mode

        // The simple box: output fps, latency, capture card, resolution.
        let (pw, ph) = processedSize.0 > 0 ? processedSize : (w, h)
        var resolution = "\(w)×\(h)"
        if (pw, ph) != (w, h) { resolution += " → \(pw)×\(ph) (processed)" }
        if upscalingActive, let upscaler, upscaler.inputWidth == pw, upscaler.inputHeight == ph {
            resolution = "\(w)×\(h) → \(upscaler.outputWidth)×\(upscaler.outputHeight) (upscaled)"
        } else if upscaleTarget > 0 {
            resolution += " (upscaling off)"
        }
        var live = LiveStats()
        live.device = videoDeviceName
        live.audio = audioDeviceName ?? "none"
        live.inputWidth = w; live.inputHeight = h
        live.processWidth = pw; live.processHeight = ph
        live.outputWidth = pw; live.outputHeight = ph
        live.inputFPS = inFps; live.gameFPS = gameFps; live.outputFPS = outFps
        live.onePercentLow = presenter.onePercentLowFPS()
        live.latencyMs = clock.delayMs
        live.screenHz = displayRefresh
        live.budgetMs = min(100, max(8, averageUniqueInterval * 1000))
        live.lateFrames = late; live.skipped = skippedCount
        live.audioPacketsPerSecond = Double(packets) / elapsed
        live.audioPlayback = clock.audioStatus
        let perGap = generator?.framesPerGap ?? 1
        if !frameGenerationOn { live.frameGeneration = "Off" }
        else if generatorFailed != nil { live.frameGeneration = "Unavailable" }
        else if attempts > 0 {
            live.frameGeneration = "\(perGap + 1)× (\(Int(gameFps.rounded())) → \(Int(outFps.rounded())) fps)"
            live.generateMs = avgMs
        } else { live.frameGeneration = "Standby (game is fast enough)" }
        if upscalingActive, let upscaler, upscaler.inputWidth == pw, upscaler.inputHeight == ph {
            live.outputWidth = upscaler.outputWidth; live.outputHeight = upscaler.outputHeight
            live.upscaling = "MetalFX → \(upscaler.outputWidth)×\(upscaler.outputHeight)"
            live.upscaleMs = upscaleAvgMs
            live.upscaledFramesPerGameFrame = attempts > 0 ? perGap + 1 : 1
        } else if upscaleFailed != nil { live.upscaling = "Unavailable" }
        else if upscaleFellBack { live.upscaling = "Off (too slow)" }
        else if upscaleTarget > 0 { live.upscaling = "Off at 120 fps" }
        if es.level != .off {
            live.enhancement = es.level.name
            if es.denoiseCount > 0 { live.cleanUpMs = es.denoiseMs / Double(es.denoiseCount) }
            if es.pictureCount > 0 { live.pictureMs = es.pictureMs / Double(es.pictureCount) }
            if es.fellBack && es.level.denoiseStrength > 0 { live.enhancement += " · clean-up off" }
            live.protectedRects = es.protectedRects
            live.subtitleStrip = Int((es.subtitleTop * 100).rounded())
        }

        let simple = String(format: "Output: %.0f fps  ·  Latency: %.0f ms", outFps, clock.delayMs)
            + "\nCapture card: \(videoDeviceName)"
            + "\nResolution: \(resolution)"

        uniqueCount = 0; outputCount = 0; generatedCount = 0; skippedCount = 0
        processingMsTotal = 0
        attempts = 0
        slowestMs = 0
        statsStart = now

        if let callback = onStats {
            let snapshot = live
            Task { @MainActor in callback(simple, text, snapshot) }
        }
    }
}

// MARK: - Mac catalog (Apple silicon)

struct MacModel {
    let family: String   // "MacBook Air", "Mac mini", …
    let name: String     // shown in the menu
    let chip: String     // "M4", "M4 Pro", "A18 Pro", …
}

enum MacCatalog {
    static let families = ["MacBook Air", "MacBook Neo", "MacBook Pro", "Mac mini", "iMac", "Mac Studio", "Mac Pro"]

    static let models: [MacModel] = [
        MacModel(family: "MacBook Air", name: "MacBook Air 13″ · M1 (2020)", chip: "M1"),
        MacModel(family: "MacBook Air", name: "MacBook Air 13″ · M2 (2022)", chip: "M2"),
        MacModel(family: "MacBook Air", name: "MacBook Air 15″ · M2 (2023)", chip: "M2"),
        MacModel(family: "MacBook Air", name: "MacBook Air 13″ · M3 (2024)", chip: "M3"),
        MacModel(family: "MacBook Air", name: "MacBook Air 15″ · M3 (2024)", chip: "M3"),
        MacModel(family: "MacBook Air", name: "MacBook Air 13″ · M4 (2025)", chip: "M4"),
        MacModel(family: "MacBook Air", name: "MacBook Air 15″ · M4 (2025)", chip: "M4"),
        MacModel(family: "MacBook Air", name: "MacBook Air 13″ · M5 (2026)", chip: "M5"),
        MacModel(family: "MacBook Air", name: "MacBook Air 15″ · M5 (2026)", chip: "M5"),

        MacModel(family: "MacBook Neo", name: "MacBook Neo · A18 Pro (2026)", chip: "A18 Pro"),

        MacModel(family: "MacBook Pro", name: "MacBook Pro 13″ · M1 (2020)", chip: "M1"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M1 Pro (2021)", chip: "M1 Pro"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M1 Max (2021)", chip: "M1 Max"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 13″ · M2 (2022)", chip: "M2"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M2 Pro (2023)", chip: "M2 Pro"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M2 Max (2023)", chip: "M2 Max"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″ · M3 (2023)", chip: "M3"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M3 Pro (2023)", chip: "M3 Pro"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M3 Max (2023)", chip: "M3 Max"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″ · M4 (2024)", chip: "M4"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M4 Pro (2024)", chip: "M4 Pro"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M4 Max (2024)", chip: "M4 Max"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″ · M5 (2025)", chip: "M5"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M5 Pro (2026)", chip: "M5 Pro"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″/16″ · M5 Max (2026)", chip: "M5 Max"),
        MacModel(family: "MacBook Pro", name: "MacBook Pro 14″ · M6 (2026)", chip: "M6"),

        MacModel(family: "Mac mini", name: "Mac mini · M1 (2020)", chip: "M1"),
        MacModel(family: "Mac mini", name: "Mac mini · M2 (2023)", chip: "M2"),
        MacModel(family: "Mac mini", name: "Mac mini · M2 Pro (2023)", chip: "M2 Pro"),
        MacModel(family: "Mac mini", name: "Mac mini · M4 (2024)", chip: "M4"),
        MacModel(family: "Mac mini", name: "Mac mini · M4 Pro (2024)", chip: "M4 Pro"),
        MacModel(family: "Mac mini", name: "Mac mini · M5 Pro (2026)", chip: "M5 Pro"),
        MacModel(family: "Mac mini", name: "Mac mini · M6 (2026)", chip: "M6"),

        MacModel(family: "iMac", name: "iMac 24″ · M1 (2021)", chip: "M1"),
        MacModel(family: "iMac", name: "iMac 24″ · M3 (2023)", chip: "M3"),
        MacModel(family: "iMac", name: "iMac 24″ · M4 (2024)", chip: "M4"),

        MacModel(family: "Mac Studio", name: "Mac Studio · M1 Max (2022)", chip: "M1 Max"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M1 Ultra (2022)", chip: "M1 Ultra"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M2 Max (2023)", chip: "M2 Max"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M2 Ultra (2023)", chip: "M2 Ultra"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M4 Max (2025)", chip: "M4 Max"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M3 Ultra (2025)", chip: "M3 Ultra"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M5 Max (2026)", chip: "M5 Max"),
        MacModel(family: "Mac Studio", name: "Mac Studio · M5 Ultra (2026)", chip: "M5 Ultra"),

        MacModel(family: "Mac Pro", name: "Mac Pro · M2 Ultra (2023)", chip: "M2 Ultra"),
    ]

    static let otherName = "Other Apple silicon Mac"

    private static func sysctlString(_ key: String) -> String {
        var size = 0
        guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(key, &buffer, &size, nil, 0) == 0 else { return "" }
        return String(cString: buffer)
    }

    /// "Apple M4 Pro" → "M4 Pro"
    static func detectedChip() -> String {
        let brand = sysctlString("machdep.cpu.brand_string")
        return brand.hasPrefix("Apple ") ? String(brand.dropFirst(6)) : brand
    }

    /// The Mac's marketing name from the system, e.g. "Mac mini (2024)", when available.
    static func productName() -> String? {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        guard service != 0 else { return nil }
        defer { IOObjectRelease(service) }
        guard let value = IORegistryEntryCreateCFProperty(service, "product-name" as CFString,
                                                          kCFAllocatorDefault, 0)?.takeRetainedValue(),
              let data = value as? Data else { return nil }
        let name = String(decoding: data.filter { $0 != 0 }, as: UTF8.self)
        return name.isEmpty ? nil : name
    }

    /// "Mac mini", "MacBook Pro", … as macOS reports it (takes about half a second, so call it off the main thread).
    static func machineNameFromSystem() -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/sbin/system_profiler")
        process.arguments = ["SPHardwareDataType", "-json"]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let items = json["SPHardwareDataType"] as? [[String: Any]],
              let name = items.first?["machine_name"] as? String else { return nil }
        return name
    }

    static func detectedFamily(machineName: String? = nil) -> String? {
        let hints = [(machineName ?? ""), (productName() ?? ""), sysctlString("hw.model")].joined(separator: " ").lowercased()
        let byKeyword: [(String, String)] = [
            ("macbook air", "MacBook Air"), ("macbookair", "MacBook Air"),
            ("macbook neo", "MacBook Neo"),
            ("macbook pro", "MacBook Pro"), ("macbookpro", "MacBook Pro"),
            ("mac mini", "Mac mini"), ("macmini", "Mac mini"),
            ("imac", "iMac"), ("mac studio", "Mac Studio"), ("mac pro", "Mac Pro"),
        ]
        return byKeyword.first(where: { hints.contains($0.0) })?.1
    }

    /// Index of the catalog entry that best matches this Mac, if any.
    static func detectedIndex(machineName: String? = nil) -> Int? {
        let chip = detectedChip()
        let sameChip = models.indices.filter { models[$0].chip.caseInsensitiveCompare(chip) == .orderedSame }
        guard !sameChip.isEmpty else { return nil }
        if let family = detectedFamily(machineName: machineName),
           let match = sameChip.last(where: { models[$0].family == family }) {
            return match
        }
        return sameChip.last
    }

    /// What to expect at 120 fps on a given chip (only an M4 Mac mini has actually been tested).
    static func advice120(for chip: String?) -> String {
        guard let chip, !chip.isEmpty else {
            return "120 fps needs a fast chip. If your Mac can't keep up, Smooth60 switches to 60 fps automatically."
        }
        if chip.contains("Ultra") || chip.contains("Max") {
            return "120 fps: your \(chip) has the graphics power for it."
        }
        if chip.hasPrefix("M") && chip.contains("Pro") {
            return "120 fps: your \(chip) should manage it. If it can't keep up, Smooth60 switches to 60 fps automatically."
        }
        return "120 fps may be too heavy for the \(chip) at 1080p (an M4 Mac mini fell just short in testing). If it can't keep up, Smooth60 switches to 60 fps automatically."
    }
}

// MARK: - Look (black + red)

@MainActor
enum Theme {
    static let window = NSColor(srgbRed: 0.027, green: 0.027, blue: 0.031, alpha: 1)
    static let panel = NSColor(srgbRed: 0.070, green: 0.070, blue: 0.078, alpha: 1)
    static let raised = NSColor(srgbRed: 0.115, green: 0.115, blue: 0.125, alpha: 1)
    static let track = NSColor(srgbRed: 0.17, green: 0.17, blue: 0.18, alpha: 1)
    static let hairline = NSColor(white: 1, alpha: 0.08)
    static let redBorder = NSColor(srgbRed: 0.42, green: 0.06, blue: 0.09, alpha: 1)
    static let red = NSColor(srgbRed: 0.91, green: 0.11, blue: 0.16, alpha: 1)
    static let redDeep = NSColor(srgbRed: 0.55, green: 0.04, blue: 0.08, alpha: 1)
    static let text = NSColor(white: 0.95, alpha: 1)
    static let muted = NSColor(white: 0.60, alpha: 1)
    static let faint = NSColor(white: 0.40, alpha: 1)

    static func font(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.systemFont(ofSize: size, weight: weight)
    }
    static func mono(_ size: CGFloat, _ weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.monospacedDigitSystemFont(ofSize: size, weight: weight)
    }
    static func label(_ text: String, _ size: CGFloat, _ weight: NSFont.Weight = .regular,
                      _ color: NSColor = Theme.text) -> NSTextField {
        let field = NSTextField(labelWithString: text)
        field.font = font(size, weight)
        field.textColor = color
        field.lineBreakMode = .byTruncatingTail
        field.translatesAutoresizingMaskIntoConstraints = false
        return field
    }
    static func wrapping(_ text: String, _ size: CGFloat, _ color: NSColor = Theme.muted, width: CGFloat) -> NSTextField {
        let field = NSTextField(wrappingLabelWithString: text)
        field.font = font(size)
        field.textColor = color
        field.preferredMaxLayoutWidth = width
        field.translatesAutoresizingMaskIntoConstraints = false
        return field
    }
    /// Small spaced capitals used for panel headings.
    static func heading(_ text: String) -> NSTextField {
        let field = label("", 10.5, .semibold, Theme.faint)
        field.attributedStringValue = NSAttributedString(string: text.uppercased(), attributes: [
            .font: font(10.5, .semibold), .foregroundColor: Theme.faint, .kern: 1.1,
        ])
        return field
    }
}

final class PanelView: NSView {
    init(fill: NSColor = Theme.panel, border: NSColor? = Theme.hairline, radius: CGFloat = 10) {
        super.init(frame: .zero)
        wantsLayer = true
        layer?.backgroundColor = fill.cgColor
        layer?.cornerRadius = radius
        if let border {
            layer?.borderColor = border.cgColor
            layer?.borderWidth = 1
        }
        translatesAutoresizingMaskIntoConstraints = false
    }
    required init?(coder: NSCoder) { fatalError("not used") }
    func setFill(_ color: NSColor) { layer?.backgroundColor = color.cgColor }
    func setBorder(_ color: NSColor) { layer?.borderColor = color.cgColor }
}

/// Red-when-on switch (system switches follow the Mac's accent colour, not ours).
final class ToggleSwitch: NSView {
    var isOn: Bool { didSet { needsDisplay = true; setAccessibilityValue(NSNumber(value: isOn)) } }
    var isEnabled = true { didSet { needsDisplay = true; setAccessibilityEnabled(isEnabled) } }
    var onChange: ((Bool) -> Void)?

    init(on: Bool, label: String) {
        isOn = on
        super.init(frame: NSRect(x: 0, y: 0, width: 40, height: 22))
        translatesAutoresizingMaskIntoConstraints = false
        widthAnchor.constraint(equalToConstant: 40).isActive = true
        heightAnchor.constraint(equalToConstant: 22).isActive = true
        setAccessibilityElement(true)
        setAccessibilityRole(.checkBox)
        setAccessibilityLabel(label)
        setAccessibilityValue(NSNumber(value: on))
    }
    required init?(coder: NSCoder) { fatalError("not used") }

    override func draw(_ dirtyRect: NSRect) {
        let r = bounds.insetBy(dx: 1, dy: 1)
        let track = NSBezierPath(roundedRect: r, xRadius: r.height / 2, yRadius: r.height / 2)
        (isOn ? (isEnabled ? Theme.red : Theme.redDeep) : Theme.track).setFill()
        track.fill()
        let d = r.height - 4
        let x = isOn ? r.maxX - d - 2 : r.minX + 2
        NSColor.white.withAlphaComponent(isEnabled ? 1 : 0.35).setFill()
        NSBezierPath(ovalIn: NSRect(x: x, y: r.minY + 2, width: d, height: d)).fill()
    }

    private func flip() {
        guard isEnabled else { return }
        isOn.toggle()
        onChange?(isOn)
    }
    override func mouseDown(with event: NSEvent) { flip() }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func accessibilityPerformPress() -> Bool { flip(); return true }
}

/// Flat rounded button used in the top bar.
final class PillButton: NSButton {
    private var label = ""
    private var labelColor = NSColor.white

    init(title: String, symbol: String?, fill: NSColor, target: AnyObject?, action: Selector?, width: CGFloat = 124) {
        super.init(frame: .zero)
        self.target = target
        self.action = action
        isBordered = false
        wantsLayer = true
        layer?.cornerRadius = 7
        layer?.borderWidth = 1
        layer?.borderColor = Theme.hairline.cgColor
        layer?.backgroundColor = fill.cgColor
        if let symbol, let image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil) {
            self.image = image.withSymbolConfiguration(.init(pointSize: 12, weight: .semibold))
            imagePosition = .imageLeading
            imageHugsTitle = true          // icon sits right next to the text, both centred
            contentTintColor = .white
        }
        alignment = .center
        imageScaling = .scaleNone
        translatesAutoresizingMaskIntoConstraints = false
        heightAnchor.constraint(equalToConstant: 30).isActive = true
        widthAnchor.constraint(equalToConstant: width).isActive = true   // top-bar buttons share one size
        setLabel(title)
    }
    required init?(coder: NSCoder) { fatalError("not used") }

    func setLabel(_ text: String, color: NSColor = .white) {
        label = text
        labelColor = color
        let centred = NSMutableParagraphStyle()
        centred.alignment = .center
        attributedTitle = NSAttributedString(string: " \(text)", attributes: [
            .foregroundColor: color, .font: Theme.font(12.5, .semibold), .paragraphStyle: centred,
        ])
        setAccessibilityLabel(text)
        invalidateIntrinsicContentSize()
    }
    func setFill(_ color: NSColor) { layer?.backgroundColor = color.cgColor }
}

/// Tab heading with a red underline when selected.
final class TabButton: NSButton {
    private let name: String
    private let underline = NSView()
    var isSelected = false { didSet { refresh() } }

    init(title: String, tag: Int, target: AnyObject?, action: Selector?) {
        name = title
        super.init(frame: .zero)
        self.tag = tag
        self.target = target
        self.action = action
        isBordered = false
        translatesAutoresizingMaskIntoConstraints = false
        underline.wantsLayer = true
        underline.layer?.backgroundColor = Theme.red.cgColor
        underline.layer?.cornerRadius = 1
        underline.translatesAutoresizingMaskIntoConstraints = false
        addSubview(underline)
        NSLayoutConstraint.activate([
            underline.heightAnchor.constraint(equalToConstant: 2),
            underline.bottomAnchor.constraint(equalTo: bottomAnchor),
            underline.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 6),
            underline.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -6),
            heightAnchor.constraint(equalToConstant: 34),
        ])
        refresh()
    }
    required init?(coder: NSCoder) { fatalError("not used") }

    private func refresh() {
        attributedTitle = NSAttributedString(string: name, attributes: [
            .foregroundColor: isSelected ? Theme.text : Theme.muted,
            .font: Theme.font(12.5, isSelected ? .semibold : .medium),
        ])
        underline.isHidden = !isSelected
    }
}

/// Horizontal bar: how much of the time budget a stage uses.
final class BarView: NSView {
    var fraction: CGFloat = 0 { didSet { needsDisplay = true } }
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        translatesAutoresizingMaskIntoConstraints = false
        heightAnchor.constraint(equalToConstant: 6).isActive = true
    }
    required init?(coder: NSCoder) { fatalError("not used") }
    override func draw(_ dirtyRect: NSRect) {
        let r = bounds
        Theme.track.setFill()
        NSBezierPath(roundedRect: r, xRadius: 3, yRadius: 3).fill()
        let f = max(0, min(1, fraction))
        guard f > 0 else { return }
        (fraction > 0.9 ? NSColor(srgbRed: 1, green: 0.35, blue: 0.35, alpha: 1) : Theme.red).setFill()
        NSBezierPath(roundedRect: NSRect(x: r.minX, y: r.minY, width: max(r.height, r.width * f), height: r.height),
                     xRadius: 3, yRadius: 3).fill()
    }
}

/// Output frame rate over the last minute.
final class SparklineView: NSView {
    var values: [Double] = [] { didSet { needsDisplay = true } }
    var target: Double = 60 { didSet { needsDisplay = true } }
    var capacity = 60

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        translatesAutoresizingMaskIntoConstraints = false
    }
    required init?(coder: NSCoder) { fatalError("not used") }

    override func draw(_ dirtyRect: NSRect) {
        let r = bounds.insetBy(dx: 1, dy: 3)
        let top = max(target * 1.2, values.max() ?? 0, 1)
        func y(_ v: Double) -> CGFloat { r.minY + CGFloat(v / top) * r.height }
        // Target line.
        let guide = NSBezierPath()
        guide.move(to: NSPoint(x: r.minX, y: y(target)))
        guide.line(to: NSPoint(x: r.maxX, y: y(target)))
        guide.setLineDash([3, 3], count: 2, phase: 0)
        Theme.faint.withAlphaComponent(0.6).setStroke()
        guide.lineWidth = 1
        guide.stroke()
        guard values.count > 1 else { return }
        let step = r.width / CGFloat(max(capacity - 1, 1))
        let startX = r.maxX - step * CGFloat(values.count - 1)
        let line = NSBezierPath()
        for (i, v) in values.enumerated() {
            let p = NSPoint(x: startX + step * CGFloat(i), y: y(v))
            if i == 0 { line.move(to: p) } else { line.line(to: p) }
        }
        let area = line.copy() as! NSBezierPath
        area.line(to: NSPoint(x: r.maxX, y: r.minY))
        area.line(to: NSPoint(x: startX, y: r.minY))
        area.close()
        NSGradient(starting: Theme.red.withAlphaComponent(0.35), ending: Theme.red.withAlphaComponent(0.0))?
            .draw(in: area, angle: -90)
        Theme.red.setStroke()
        line.lineWidth = 1.6
        line.lineJoinStyle = .round
        line.stroke()
    }
}

/// Round gauge with the value in the middle.
final class RingGauge: NSView {
    var fraction: CGFloat = 0 { didSet { needsDisplay = true } }
    var valueText = "–" { didSet { needsDisplay = true; setAccessibilityValue(valueText) } }

    init(size: CGFloat, label: String) {
        super.init(frame: NSRect(x: 0, y: 0, width: size, height: size))
        translatesAutoresizingMaskIntoConstraints = false
        widthAnchor.constraint(equalToConstant: size).isActive = true
        heightAnchor.constraint(equalToConstant: size).isActive = true
        setAccessibilityElement(true)
        setAccessibilityRole(.levelIndicator)
        setAccessibilityLabel(label)
    }
    required init?(coder: NSCoder) { fatalError("not used") }

    override func draw(_ dirtyRect: NSRect) {
        let lineWidth: CGFloat = 5
        let r = bounds.insetBy(dx: lineWidth / 2 + 1, dy: lineWidth / 2 + 1)
        let centre = NSPoint(x: r.midX, y: r.midY)
        let radius = min(r.width, r.height) / 2
        let track = NSBezierPath()
        track.appendArc(withCenter: centre, radius: radius, startAngle: 0, endAngle: 360)
        track.lineWidth = lineWidth
        Theme.track.setStroke()
        track.stroke()
        let f = max(0, min(1, fraction))
        if f > 0 {
            let arc = NSBezierPath()
            arc.appendArc(withCenter: centre, radius: radius, startAngle: 90, endAngle: 90 - 360 * f, clockwise: true)
            arc.lineWidth = lineWidth
            arc.lineCapStyle = .round
            Theme.red.setStroke()
            arc.stroke()
        }
        let attributes: [NSAttributedString.Key: Any] = [.font: Theme.mono(11.5, .semibold), .foregroundColor: Theme.text]
        let size = (valueText as NSString).size(withAttributes: attributes)
        (valueText as NSString).draw(at: NSPoint(x: centre.x - size.width / 2, y: centre.y - size.height / 2),
                                     withAttributes: attributes)
    }
}

// MARK: - App (one window: top bar, video, settings panel, live stats)

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate, NSWindowDelegate {
    private let defaults = UserDefaults.standard
    private var window: NSWindow!
    private var capture: CaptureController!
    private var running = false
    private var currentFPS = 60
    private var currentDeviceID: String?
    private var recordingStarted: Date?
    private var recordTimer: Timer?
    private var detectedIndex = MacCatalog.detectedIndex()
    private let upscaleTargets = Upscaler.targets(forInputHeight: 1080)

    // Layout
    private var titleStrip: NSTextField!
    private var topBar: PanelView!
    private var sidebar: PanelView!
    private var bottomRow: NSView!
    private var videoPanel: PanelView!
    private var normalConstraints: [NSLayoutConstraint] = []
    private var fullScreenConstraints: [NSLayoutConstraint] = []

    // Video overlays
    private var videoView: VideoView!
    private var statsBox: NSView!
    private var statsLabel: NSTextField!
    private var noticeLabel: NSTextField!
    private var recBadge: NSTextField!
    private var videoMessage: NSTextField!

    // Stream output window (a clean video-only window for Streamlabs / OBS window capture)
    private var clock: PlaybackClock!
    private var streamWindow: NSWindow?
    private var streamView: VideoView?
    private var streamSwitch: ToggleSwitch!
    private var streamItem: NSMenuItem!
    private var lastOutputSize = (1920, 1080)

    // Top bar
    private let cardChip = Theme.label("", 12, .medium)
    private let signalChip = Theme.label("", 12, .medium, Theme.muted)
    private let outputChip = Theme.label("", 12, .medium, Theme.muted)
    private var startButton: PillButton!
    private var recordButton: PillButton!
    private var shotButton: PillButton!
    private var compareButton: PillButton!
    private var soundButton: PillButton!
    private var soundSwitch: ToggleSwitch!
    private let audioOutputPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let volumeSlider = NSSlider(value: 1, minValue: 0, maxValue: 1, target: nil, action: nil)
    private let syncSlider = NSSlider(value: 0, minValue: -60, maxValue: 150, target: nil, action: nil)
    private let syncValue = Theme.label("0 ms", 11.5, .medium)
    private let audioHint = Theme.wrapping("", 11.5, width: 320)
    private var soundItem: NSMenuItem!
    private var compareItem: NSMenuItem!

    // Settings panel
    private var tabButtons: [TabButton] = []
    private let tabView = NSTabView()
    private let macPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let cardPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let fpsPopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let upscalePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let enhancePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let rangePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private var frameGenSwitch: ToggleSwitch!
    private var hudSwitch: ToggleSwitch!
    private let subtitlePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private var protectedSwitch: ToggleSwitch!
    private var overlaySwitch: ToggleSwitch!
    private var detailedSwitch: ToggleSwitch!
    private let sourceHint = Theme.wrapping("", 11.5, width: 320)
    private let frameGenHint = Theme.wrapping("", 11.5, width: 320)
    private let qualityHint = Theme.wrapping("", 11.5, width: 320)
    private let pipelineLabel = Theme.wrapping("", 11, Theme.faint, width: 330)
    private let recordingsPathLabel = Theme.wrapping("", 11, Theme.muted, width: 190)
    private let screenshotsPathLabel = Theme.wrapping("", 11, Theme.muted, width: 190)

    // Live stats panels
    private var signalValues: [NSTextField] = []
    private var stageBars: [BarView] = []
    private var stageValues: [NSTextField] = []
    private let budgetLabel = Theme.label("", 10.5, .medium, Theme.faint)
    private let fpsBig = Theme.label("–", 30, .bold, Theme.red)
    private let fpsSub = Theme.label("", 11.5, .regular, Theme.muted)
    private let graph = SparklineView(frame: .zero)
    private let latencyGauge = RingGauge(size: 54, label: "Latency")
    private let processingGauge = RingGauge(size: 54, label: "Processing")
    private let loadGauge = RingGauge(size: 54, label: "Load")
    private var fpsHistory: [Double] = []

    // Menus
    private let inputMenu = NSMenu(title: "Input")
    private var startItem: NSMenuItem!
    private var recordItem: NSMenuItem!
    private var frameGenItem: NSMenuItem!
    private var statsItem: NSMenuItem!
    private var detailedStatsItem: NSMenuItem!
    private var rangeItems: [NSMenuItem] = []
    private var enhanceItems: [NSMenuItem] = []
    private var hudItem: NSMenuItem!

    // MARK: Saved settings

    private var savedFPS: Int { defaults.integer(forKey: "targetFPS") == 120 ? 120 : 60 }
    private var frameGenOn: Bool { defaults.object(forKey: "frameGeneration") as? Bool ?? true }
    private var savedUpscale: Int { defaults.integer(forKey: "upscaleTag") }
    private var savedEnhance: EnhanceLevel { EnhanceLevel(rawValue: defaults.integer(forKey: "enhanceLevel")) ?? .off }
    private var savedHUD: Bool { defaults.object(forKey: "protectHUD") as? Bool ?? true }
    private var savedSubtitles: SubtitleMode { SubtitleMode(rawValue: defaults.integer(forKey: "subtitleMode")) ?? .auto }
    private var showProtected: Bool { defaults.bool(forKey: "showProtected") }
    private var savedRange: ColourRange { ColourRange(rawValue: defaults.integer(forKey: "colourRange")) ?? .auto }
    private var showOverlay: Bool { defaults.bool(forKey: "showOverlay") }
    private var detailedOverlay: Bool { defaults.bool(forKey: "detailedStats") }
    private var soundOn: Bool { defaults.object(forKey: "soundOn") as? Bool ?? true }
    private var savedVolume: Double { defaults.object(forKey: "soundVolume") as? Double ?? 1 }
    private var savedAudioSync: Double { defaults.double(forKey: "audioSyncMs") }
    private var savedAudioOutput: String? { defaults.string(forKey: "audioOutputUID") }

    // MARK: Launch

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenus()
        buildWindow()

        clock = PlaybackClock(displayLayer: videoView.displayLayer, delayMs: 40)
        capture = CaptureController(presenter: Presenter(renderer: videoView.displayLayer.sampleBufferRenderer),
                                    clock: clock)
        capture.onStats = { [weak self] simple, detailed, live in
            self?.updateStats(simple: simple, detailed: detailed, live: live)
        }
        capture.onNotice = { [weak self] text in self?.showNotice(text) }
        updateDisplayRefresh()
        NotificationCenter.default.addObserver(forName: NSWindow.didChangeScreenNotification, object: window,
                                               queue: .main) { [weak self] _ in
            Task { @MainActor in self?.updateDisplayRefresh() }
        }
        NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.updateDisplayRefresh() }
        }

        // F7 = screenshot, F8 = record (like Veyra). On Mac keyboards these may need the fn key.
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            let code = event.keyCode
            guard code == 98 || code == 100 else { return event }   // F7, F8
            let handled: Bool = MainActor.assumeIsolated {
                guard let self, self.running else { return false }
                if code == 98 { self.takeScreenshot() } else { self.toggleRecording() }
                return true
            }
            return handled ? nil : event
        }

        refreshDevices()
        refreshAudioOutputs()
        applyAudio()
        syncControls()
        updateFolderLabels()
        updateStartButton()
        showIdle()
        window.makeKeyAndOrderFront(nil)
        if defaults.bool(forKey: "streamOutput") { setStreamOutput(true) }
        NSApp.activate(ignoringOtherApps: true)
        detectMacInBackground()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
    func applicationWillTerminate(_ notification: Notification) { capture?.stop() }

    private func updateDisplayRefresh() {
        capture.setDisplayRefresh(window.screen?.maximumFramesPerSecond ?? NSScreen.main?.maximumFramesPerSecond ?? 60)
    }

    // MARK: Window

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1500, height: 920),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                          backing: .buffered, defer: false)
        window.title = "Smooth60"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.appearance = NSAppearance(named: .darkAqua)
        window.backgroundColor = Theme.window
        window.minSize = NSSize(width: 1180, height: 760)
        window.collectionBehavior = [.fullScreenPrimary]
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.setFrameAutosaveName("Smooth60Main")

        let content = NSView()
        content.wantsLayer = true
        content.layer?.backgroundColor = Theme.window.cgColor
        window.contentView = content

        titleStrip = Theme.label("", 12, .bold, Theme.red)
        titleStrip.attributedStringValue = NSAttributedString(string: "SMOOTH60", attributes: [
            .font: Theme.font(12, .heavy), .foregroundColor: Theme.red, .kern: 2.5,
        ])
        content.addSubview(titleStrip)

        topBar = buildTopBar()
        sidebar = buildSidebar()
        bottomRow = buildBottomRow()
        videoPanel = buildVideoPanel()
        for view in [topBar!, sidebar!, bottomRow!, videoPanel!] as [NSView] { content.addSubview(view) }

        let margin: CGFloat = 10
        NSLayoutConstraint.activate([
            titleStrip.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            titleStrip.topAnchor.constraint(equalTo: content.topAnchor, constant: 7),
        ])
        normalConstraints = [
            topBar.topAnchor.constraint(equalTo: content.topAnchor, constant: 30),
            topBar.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: margin),
            topBar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -margin),
            topBar.heightAnchor.constraint(equalToConstant: 46),

            sidebar.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: margin),
            sidebar.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -margin),
            sidebar.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -margin),
            sidebar.widthAnchor.constraint(equalToConstant: 372),

            bottomRow.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: margin),
            bottomRow.trailingAnchor.constraint(equalTo: sidebar.leadingAnchor, constant: -margin),
            bottomRow.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -margin),
            bottomRow.heightAnchor.constraint(equalToConstant: 178),

            videoPanel.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: margin),
            videoPanel.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: margin),
            videoPanel.trailingAnchor.constraint(equalTo: sidebar.leadingAnchor, constant: -margin),
            videoPanel.bottomAnchor.constraint(equalTo: bottomRow.topAnchor, constant: -margin),
        ]
        fullScreenConstraints = [
            videoPanel.topAnchor.constraint(equalTo: content.topAnchor),
            videoPanel.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            videoPanel.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            videoPanel.bottomAnchor.constraint(equalTo: content.bottomAnchor),
        ]
        NSLayoutConstraint.activate(normalConstraints)
        window.center()
    }

    private func chip(_ label: NSTextField, dot: Bool = false) -> NSView {
        let box = PanelView(fill: Theme.raised, border: Theme.hairline, radius: 6)
        var views: [NSView] = []
        if dot {
            let d = PanelView(fill: Theme.red, border: nil, radius: 3.5)
            d.widthAnchor.constraint(equalToConstant: 7).isActive = true
            d.heightAnchor.constraint(equalToConstant: 7).isActive = true
            views.append(d)
        }
        views.append(label)
        let row = NSStackView(views: views)
        row.spacing = 7
        row.alignment = .centerY
        row.translatesAutoresizingMaskIntoConstraints = false
        box.addSubview(row)
        NSLayoutConstraint.activate([
            row.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 10),
            row.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -10),
            row.centerYAnchor.constraint(equalTo: box.centerYAnchor),
            box.heightAnchor.constraint(equalToConstant: 28),
        ])
        label.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        return box
    }

    private func buildTopBar() -> PanelView {
        let bar = PanelView()
        startButton = PillButton(title: "Start", symbol: "play.fill", fill: Theme.red, target: self, action: #selector(startStopPressed))
        recordButton = PillButton(title: "Record", symbol: "record.circle", fill: Theme.raised, target: self, action: #selector(toggleRecording))
        shotButton = PillButton(title: "Screenshot", symbol: "camera", fill: Theme.raised, target: self, action: #selector(takeScreenshot))
        recordButton.contentTintColor = Theme.red
        soundButton = PillButton(title: "Sound", symbol: "speaker.wave.2.fill", fill: Theme.raised,
                                 target: self, action: #selector(toggleSound), width: 96)
        compareButton = PillButton(title: "Before / After", symbol: "rectangle.split.2x1", fill: Theme.raised,
                                   target: self, action: #selector(toggleCompare))
        streamSwitch = ToggleSwitch(on: false, label: "Stream output window")
        streamSwitch.onChange = { [weak self] on in self?.setStreamOutput(on) }
        let streamLabel = Theme.label("Stream output", 12.5, .semibold, Theme.red)
        let streamGroup = NSStackView(views: [streamLabel, streamSwitch])
        streamGroup.orientation = .horizontal
        streamGroup.alignment = .centerY
        streamGroup.spacing = 8

        let left = NSStackView(views: [chip(cardChip, dot: true), chip(signalChip), chip(outputChip)])
        left.spacing = 8
        let right = NSStackView(views: [streamGroup, soundButton, compareButton, shotButton, recordButton, startButton])
        right.spacing = 8
        right.setCustomSpacing(16, after: streamGroup)
        for stack in [left, right] {
            stack.orientation = .horizontal
            stack.alignment = .centerY
            stack.translatesAutoresizingMaskIntoConstraints = false
            bar.addSubview(stack)
        }
        NSLayoutConstraint.activate([
            left.leadingAnchor.constraint(equalTo: bar.leadingAnchor, constant: 9),
            left.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            right.trailingAnchor.constraint(equalTo: bar.trailingAnchor, constant: -9),
            right.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
            left.trailingAnchor.constraint(lessThanOrEqualTo: right.leadingAnchor, constant: -12),
        ])
        return bar
    }

    private func buildVideoPanel() -> PanelView {
        let panel = PanelView(fill: .black, border: Theme.hairline, radius: 10)
        panel.layer?.masksToBounds = true
        videoView = VideoView(frame: .zero)
        videoView.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(videoView)

        // Optional stats box over the video (off by default — the panels below show the numbers).
        statsBox = NSView()
        statsBox.wantsLayer = true
        statsBox.layer?.backgroundColor = NSColor(srgbRed: 0.80, green: 0.08, blue: 0.10, alpha: 0.92).cgColor
        statsBox.layer?.borderColor = NSColor.white.cgColor
        statsBox.layer?.borderWidth = 2
        statsBox.layer?.cornerRadius = 8
        statsBox.translatesAutoresizingMaskIntoConstraints = false
        statsBox.isHidden = !showOverlay
        videoView.addSubview(statsBox)
        statsLabel = NSTextField(labelWithString: "")
        statsLabel.font = NSFont(name: "Times New Roman", size: 15) ?? NSFont.systemFont(ofSize: 13)
        statsLabel.textColor = .white
        statsLabel.alignment = .center
        statsLabel.maximumNumberOfLines = 9
        statsLabel.translatesAutoresizingMaskIntoConstraints = false
        statsBox.addSubview(statsLabel)

        recBadge = Theme.label("", 13, .bold)
        recBadge.font = Theme.mono(13, .bold)
        recBadge.backgroundColor = Theme.red
        recBadge.drawsBackground = true
        recBadge.isHidden = true
        videoView.addSubview(recBadge)

        noticeLabel = Theme.label("", 14, .semibold)
        noticeLabel.backgroundColor = NSColor(white: 0.06, alpha: 0.92)
        noticeLabel.drawsBackground = true
        noticeLabel.alignment = .center
        noticeLabel.isHidden = true
        videoView.addSubview(noticeLabel)

        videoMessage = Theme.wrapping("", 15, Theme.muted, width: 520)
        videoMessage.alignment = .center
        videoView.addSubview(videoMessage)

        NSLayoutConstraint.activate([
            videoView.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            videoView.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            videoView.topAnchor.constraint(equalTo: panel.topAnchor),
            videoView.bottomAnchor.constraint(equalTo: panel.bottomAnchor),
            statsBox.centerXAnchor.constraint(equalTo: videoView.centerXAnchor),
            statsBox.topAnchor.constraint(equalTo: videoView.topAnchor, constant: 12),
            statsBox.widthAnchor.constraint(lessThanOrEqualTo: videoView.widthAnchor, constant: -24),
            statsLabel.leadingAnchor.constraint(equalTo: statsBox.leadingAnchor, constant: 16),
            statsLabel.trailingAnchor.constraint(equalTo: statsBox.trailingAnchor, constant: -16),
            statsLabel.topAnchor.constraint(equalTo: statsBox.topAnchor, constant: 10),
            statsLabel.bottomAnchor.constraint(equalTo: statsBox.bottomAnchor, constant: -10),
            recBadge.leadingAnchor.constraint(equalTo: videoView.leadingAnchor, constant: 14),
            recBadge.topAnchor.constraint(equalTo: videoView.topAnchor, constant: 14),
            noticeLabel.centerXAnchor.constraint(equalTo: videoView.centerXAnchor),
            noticeLabel.bottomAnchor.constraint(equalTo: videoView.bottomAnchor, constant: -24),
            videoMessage.centerXAnchor.constraint(equalTo: videoView.centerXAnchor),
            videoMessage.centerYAnchor.constraint(equalTo: videoView.centerYAnchor),
        ])
        return panel
    }

    // MARK: Settings panel

    private func row(_ title: String, _ detail: String, _ control: NSView?, detailLabel given: NSTextField? = nil) -> NSView {
        let container = NSView()
        container.translatesAutoresizingMaskIntoConstraints = false
        let titleLabel = Theme.label(title, 13, .semibold)
        let detailLabel = given ?? Theme.wrapping(detail, 11, Theme.muted, width: control == nil ? 330 : 165)
        if given != nil { detailLabel.stringValue = detail }
        let texts = NSStackView(views: [titleLabel, detailLabel])
        texts.orientation = .vertical
        texts.alignment = .leading
        texts.spacing = 3
        texts.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(texts)
        let line = PanelView(fill: Theme.hairline, border: nil, radius: 0)
        container.addSubview(line)
        var constraints = [
            texts.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 16),
            texts.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            texts.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -12),
            line.heightAnchor.constraint(equalToConstant: 1),
            line.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 16),
            line.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -16),
            line.bottomAnchor.constraint(equalTo: container.bottomAnchor),
        ]
        if let control {
            control.translatesAutoresizingMaskIntoConstraints = false
            container.addSubview(control)
            constraints += [
                control.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -16),
                control.centerYAnchor.constraint(equalTo: texts.centerYAnchor),
                texts.trailingAnchor.constraint(lessThanOrEqualTo: control.leadingAnchor, constant: -12),
                control.topAnchor.constraint(greaterThanOrEqualTo: container.topAnchor, constant: 8),
                control.bottomAnchor.constraint(lessThanOrEqualTo: container.bottomAnchor, constant: -8),
            ]
            if control is NSPopUpButton {
                constraints.append(control.widthAnchor.constraint(equalToConstant: 168))
            }
        } else {
            constraints.append(texts.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -16))
        }
        NSLayoutConstraint.activate(constraints)
        return container
    }

    private func hintRow(_ label: NSTextField) -> NSView {
        let container = NSView()
        container.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 16),
            label.trailingAnchor.constraint(lessThanOrEqualTo: container.trailingAnchor, constant: -16),
            label.topAnchor.constraint(equalTo: container.topAnchor, constant: 12),
            label.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -8),
        ])
        return container
    }

    private func page(_ rows: [NSView]) -> NSTabViewItem {
        let stack = NSStackView(views: rows)
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 0
        stack.distribution = .fill
        stack.setHuggingPriority(.required, for: .vertical)
        stack.translatesAutoresizingMaskIntoConstraints = false
        let holder = NSView()
        holder.addSubview(stack)
        var constraints = [
            stack.topAnchor.constraint(equalTo: holder.topAnchor, constant: 4),
            stack.leadingAnchor.constraint(equalTo: holder.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: holder.trailingAnchor),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: holder.bottomAnchor),
        ]
        for r in rows { constraints.append(r.widthAnchor.constraint(equalTo: stack.widthAnchor)) }
        NSLayoutConstraint.activate(constraints)
        let item = NSTabViewItem()
        item.view = holder
        return item
    }

    private func buildSidebar() -> PanelView {
        let panel = PanelView()

        // Controls
        buildMacMenu()
        macPopup.target = self; macPopup.action = #selector(macChanged)
        cardPopup.target = self; cardPopup.action = #selector(cardChanged)
        cardPopup.menu?.delegate = self
        fpsPopup.addItems(withTitles: ["60 fps  (2×)", "120 fps  (4×)"])
        fpsPopup.item(at: 0)?.tag = 60
        fpsPopup.item(at: 1)?.tag = 120
        fpsPopup.target = self; fpsPopup.action = #selector(fpsChanged)
        upscalePopup.addItem(withTitle: "Off"); upscalePopup.lastItem?.tag = 0
        for target in upscaleTargets {
            let (w, h) = Upscaler.outputSize(width: 1920, height: 1080, targetHeight: target)
            upscalePopup.addItem(withTitle: "\(h == 2160 ? "4K" : "\(h)p")  (\(w)×\(h))")
            upscalePopup.lastItem?.tag = target
        }
        if upscaleTargets.isEmpty {
            upscalePopup.addItem(withTitle: "Not supported")
            upscalePopup.lastItem?.isEnabled = false
            upscalePopup.lastItem?.tag = -1
        }
        upscalePopup.autoenablesItems = false
        upscalePopup.target = self; upscalePopup.action = #selector(upscaleChanged)
        for level in [EnhanceLevel.off, .light, .medium, .strong] {
            enhancePopup.addItem(withTitle: level.name)
            enhancePopup.lastItem?.tag = level.rawValue
        }
        enhancePopup.target = self; enhancePopup.action = #selector(enhanceChanged)
        for range in [ColourRange.auto, .limited, .full] {
            rangePopup.addItem(withTitle: range.title)
            rangePopup.lastItem?.tag = range.rawValue
        }
        rangePopup.target = self; rangePopup.action = #selector(rangeChanged)
        frameGenSwitch = ToggleSwitch(on: frameGenOn, label: "Frame generation")
        frameGenSwitch.onChange = { [weak self] on in self?.setFrameGeneration(on) }
        for mode in [SubtitleMode.auto, .normal, .large, .largest] {
            subtitlePopup.addItem(withTitle: mode.title)
            subtitlePopup.lastItem?.tag = mode.rawValue
        }
        subtitlePopup.target = self; subtitlePopup.action = #selector(subtitleModeChanged)
        protectedSwitch = ToggleSwitch(on: showProtected, label: "Show protected areas")
        protectedSwitch.onChange = { [weak self] on in
            self?.defaults.set(on, forKey: "showProtected")
            self?.syncControls()
        }
        hudSwitch = ToggleSwitch(on: savedHUD, label: "Protect HUD and subtitles")
        hudSwitch.onChange = { [weak self] on in
            self?.defaults.set(on, forKey: "protectHUD")
            self?.applyEnhancement()
        }
        overlaySwitch = ToggleSwitch(on: showOverlay, label: "Stats box over the video")
        overlaySwitch.onChange = { [weak self] on in self?.setOverlay(on) }
        detailedSwitch = ToggleSwitch(on: detailedOverlay, label: "Detailed stats")
        detailedSwitch.onChange = { [weak self] on in
            self?.defaults.set(on, forKey: "detailedStats")
            self?.syncControls()
        }
        let refreshButton = NSButton(title: "Rescan", target: self, action: #selector(rescanDevices))
        refreshButton.bezelStyle = .push
        refreshButton.controlSize = .small
        let fullScreenButton = NSButton(title: "Full screen", target: self, action: #selector(goFullScreen))
        fullScreenButton.bezelStyle = .push
        func folderButtons(change: Selector, open: Selector) -> NSView {
            let changeButton = NSButton(title: "Change…", target: self, action: change)
            changeButton.bezelStyle = .push
            let openButton = NSButton(title: "Open", target: self, action: open)
            openButton.bezelStyle = .push
            let stack = NSStackView(views: [changeButton, openButton])
            stack.orientation = .vertical
            stack.alignment = .trailing
            stack.spacing = 4
            return stack
        }
        let recordingsButton = folderButtons(change: #selector(changeRecordingsFolder), open: #selector(openRecordings))
        let shotsButton = folderButtons(change: #selector(changeScreenshotsFolder), open: #selector(openScreenshots))

        soundSwitch = ToggleSwitch(on: soundOn, label: "Game sound")
        soundSwitch.onChange = { [weak self] on in self?.setSound(on) }
        audioOutputPopup.target = self; audioOutputPopup.action = #selector(audioOutputChanged)
        audioOutputPopup.menu?.delegate = self
        volumeSlider.target = self; volumeSlider.action = #selector(volumeChanged)
        volumeSlider.translatesAutoresizingMaskIntoConstraints = false
        volumeSlider.widthAnchor.constraint(equalToConstant: 150).isActive = true
        syncSlider.target = self; syncSlider.action = #selector(audioSyncChanged)
        syncSlider.numberOfTickMarks = 22
        syncSlider.allowsTickMarkValuesOnly = true
        syncSlider.translatesAutoresizingMaskIntoConstraints = false
        syncSlider.widthAnchor.constraint(equalToConstant: 108).isActive = true
        syncValue.font = Theme.mono(11.5, .medium)
        syncValue.alignment = .right
        syncValue.widthAnchor.constraint(equalToConstant: 48).isActive = true
        let syncGroup = NSStackView(views: [syncSlider, syncValue])
        syncGroup.orientation = .horizontal
        syncGroup.spacing = 4
        let syncReset = NSButton(title: "Reset", target: self, action: #selector(resetAudioSync))
        syncReset.bezelStyle = .push
        syncReset.controlSize = .small

        let pages: [(String, [NSView])] = [
            ("Source", [
                row("Capture card", "The USB capture card your PS5 is plugged into. Game audio comes from the same card.", cardPopup),
                row("Find devices", "Plugged a card in after opening Smooth60? Scan again.", refreshButton),
                row("Your Mac", "Used for performance advice. Detected automatically.", macPopup),
                hintRow(sourceHint),
            ]),
            ("Frame Gen", [
                row("Frame generation", "Creates new frames between the game's real ones with Apple's frame interpolation. ⌘G", frameGenSwitch),
                row("Frame rate", "60 fps adds 1 frame per real frame, 120 fps adds 3. 120 falls back to 60 automatically if the Mac can't keep up.", fpsPopup),
                row("Only when needed", "Frame generation pauses by itself when the game already runs above ~40 fps.", nil),
                hintRow(frameGenHint),
            ]),
            ("Quality", [
                row("Upscaling", "MetalFX, Apple's game upscaler. 60 fps only. Turns itself off if the Mac can't keep up.", upscalePopup),
                row("Enhancement", "Light: sharpen + colour. Medium / Strong: also clean up capture noise and shimmer. ⌥⌘0–3", enhancePopup),
                row("Protect HUD & subtitles", "Leaves the corners and the subtitle area untouched so text and minimaps stay crisp.", hudSwitch),
                row("Subtitle area", "Auto finds the captions on screen and grows the protected strip to fit two-line and large captions. Or pick a fixed height.", subtitlePopup),
                row("Show protected areas", "Draws dashed outlines over the video where enhancement is held back (not in recordings or the stream).", protectedSwitch),
                hintRow(qualityHint),
            ]),
            ("Audio", [
                row("Game sound", "Plays the capture card's sound from Smooth60, in sync with the smoothed picture. No Streamlabs needed.", soundSwitch),
                row("Output", "Where the sound plays: speakers, headphones, AirPods, or the monitor.", audioOutputPopup),
                row("Volume", "Smooth60's own volume (the Mac's volume keys still apply).", volumeSlider),
                row("Sync adjust", "Sound ahead of the picture → move right. Sound late → move left. Usually 0.", syncGroup),
                row("Reset sync", "Back to 0 ms.", syncReset),
                hintRow(audioHint),
            ]),
            ("Colour", [
                row("Colour range", "Auto reads it from the card. Grey blacks → Limited. Crushed shadows → Full.", rangePopup),
                row("PS5 tip", "Set PS5 Settings → Screen and Video → RGB Range to the same value as here for the most accurate picture.", nil),
            ]),
            ("Display", [
                row("Stats box over video", "The red box at the top of the picture. ⌘I", overlaySwitch),
                row("Detailed stats", "Show every number in the stats box, for troubleshooting. ⌥⌘I", detailedSwitch),
                row("Full screen", "Video only, without the panels. ⌃⌘F", fullScreenButton),
                row("Stream output", "Turn on with the switch in the top bar. Opens a clean video-only window called “Smooth60 Stream Output”. In Streamlabs or OBS add Window Capture and pick it. ⇧⌘O", nil),
                row("Recordings", "", recordingsButton, detailLabel: recordingsPathLabel),
                row("Screenshots", "", shotsButton, detailLabel: screenshotsPathLabel),
            ]),
        ]

        tabView.tabViewType = .noTabsNoBorder
        tabView.drawsBackground = false
        tabView.translatesAutoresizingMaskIntoConstraints = false
        var buttons: [NSView] = []
        for (index, (title, rows)) in pages.enumerated() {
            tabView.addTabViewItem(page(rows))
            let button = TabButton(title: title, tag: index, target: self, action: #selector(tabPressed(_:)))
            tabButtons.append(button)
            buttons.append(button)
        }
        let tabs = NSStackView(views: buttons)
        tabs.orientation = .horizontal
        tabs.distribution = .fillEqually
        tabs.spacing = 0
        tabs.translatesAutoresizingMaskIntoConstraints = false
        let tabLine = PanelView(fill: Theme.hairline, border: nil, radius: 0)

        let reset = NSButton(title: "Reset settings", target: self, action: #selector(resetSettings))
        reset.isBordered = false
        reset.contentTintColor = Theme.muted
        reset.font = Theme.font(11, .medium)
        reset.translatesAutoresizingMaskIntoConstraints = false
        let footerLine = PanelView(fill: Theme.hairline, border: nil, radius: 0)
        let dot = PanelView(fill: Theme.red, border: nil, radius: 3)

        for view in [tabs, tabLine, tabView, footerLine, dot, pipelineLabel, reset] as [NSView] { panel.addSubview(view) }
        NSLayoutConstraint.activate([
            tabs.topAnchor.constraint(equalTo: panel.topAnchor, constant: 6),
            tabs.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 8),
            tabs.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -8),
            tabLine.topAnchor.constraint(equalTo: tabs.bottomAnchor),
            tabLine.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            tabLine.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            tabLine.heightAnchor.constraint(equalToConstant: 1),
            tabView.topAnchor.constraint(equalTo: tabLine.bottomAnchor, constant: 4),
            tabView.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            tabView.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            tabView.bottomAnchor.constraint(equalTo: footerLine.topAnchor, constant: -6),
            footerLine.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            footerLine.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            footerLine.heightAnchor.constraint(equalToConstant: 1),
            footerLine.bottomAnchor.constraint(equalTo: pipelineLabel.topAnchor, constant: -10),
            dot.widthAnchor.constraint(equalToConstant: 6),
            dot.heightAnchor.constraint(equalToConstant: 6),
            dot.leadingAnchor.constraint(equalTo: panel.leadingAnchor, constant: 14),
            dot.topAnchor.constraint(equalTo: pipelineLabel.topAnchor, constant: 4),
            pipelineLabel.leadingAnchor.constraint(equalTo: dot.trailingAnchor, constant: 8),
            pipelineLabel.trailingAnchor.constraint(lessThanOrEqualTo: panel.trailingAnchor, constant: -14),
            pipelineLabel.bottomAnchor.constraint(equalTo: reset.topAnchor, constant: -6),
            reset.trailingAnchor.constraint(equalTo: panel.trailingAnchor, constant: -12),
            reset.bottomAnchor.constraint(equalTo: panel.bottomAnchor, constant: -8),
        ])
        let savedTab = min(max(defaults.integer(forKey: "settingsTab"), 0), pages.count - 1)
        selectTab(savedTab)
        return panel
    }

    private func buildMacMenu() {
        let menu = NSMenu()
        for family in MacCatalog.families {
            let header = NSMenuItem(title: family, action: nil, keyEquivalent: "")
            header.isEnabled = false
            menu.addItem(header)
            for (index, model) in MacCatalog.models.enumerated() where model.family == family {
                let item = NSMenuItem(title: model.name + (index == detectedIndex ? "  — this Mac" : ""), action: nil, keyEquivalent: "")
                item.indentationLevel = 1
                item.tag = index
                menu.addItem(item)
            }
        }
        menu.addItem(.separator())
        let other = NSMenuItem(title: MacCatalog.otherName, action: nil, keyEquivalent: "")
        other.tag = -1
        menu.addItem(other)
        menu.autoenablesItems = false
        macPopup.menu = menu
        let initial = (defaults.object(forKey: "macModelIndex") as? Int) ?? detectedIndex ?? -1
        if let item = menu.items.first(where: { $0.isEnabled && !$0.isSeparatorItem && $0.tag == initial }) {
            macPopup.select(item)
        } else {
            macPopup.select(other)
        }
    }

    private func detectMacInBackground() {
        DispatchQueue.global(qos: .userInitiated).async {
            let index = MacCatalog.detectedIndex(machineName: MacCatalog.machineNameFromSystem())
            Task { @MainActor in
                guard index != self.detectedIndex else { return }
                self.detectedIndex = index
                for item in self.macPopup.itemArray where item.isEnabled && !item.isSeparatorItem && item.tag >= 0 {
                    item.title = MacCatalog.models[item.tag].name + (item.tag == index ? "  — this Mac" : "")
                }
                if self.defaults.object(forKey: "macModelIndex") == nil, let index,
                   let item = self.macPopup.itemArray.first(where: { $0.isEnabled && !$0.isSeparatorItem && $0.tag == index }) {
                    self.macPopup.select(item)
                }
                self.syncControls()
            }
        }
    }

    @objc private func tabPressed(_ sender: TabButton) { selectTab(sender.tag) }

    private func selectTab(_ index: Int) {
        tabView.selectTabViewItem(at: index)
        for button in tabButtons { button.isSelected = button.tag == index }
        defaults.set(index, forKey: "settingsTab")
    }

    // MARK: Live stats panels

    private func pair(_ key: String) -> (NSView, NSTextField) {
        let keyLabel = Theme.label(key, 11.5, .regular, Theme.muted)
        let value = Theme.label("–", 11.5, .medium)
        value.font = Theme.mono(11.5, .medium)
        value.alignment = .right
        value.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        let row = NSStackView(views: [keyLabel, value])
        row.orientation = .horizontal
        row.distribution = .fill
        row.spacing = 10
        keyLabel.setContentHuggingPriority(.required, for: .horizontal)
        row.translatesAutoresizingMaskIntoConstraints = false
        return (row, value)
    }

    private func column(_ views: [NSView], in panel: NSView, spacing: CGFloat) -> NSStackView {
        let stack = NSStackView(views: views)
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = spacing
        stack.translatesAutoresizingMaskIntoConstraints = false
        panel.addSubview(stack)
        for v in views { v.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true }
        return stack
    }

    private func buildBottomRow() -> NSView {
        let row = NSView()
        row.translatesAutoresizingMaskIntoConstraints = false

        // 1. Signal
        let signal = PanelView()
        var signalRows: [NSView] = [Theme.heading("Signal")]
        for key in ["Capture card", "Input", "Game", "Output", "Frame generation", "Upscaling", "Enhancement", "Screen"] {
            let (r, value) = pair(key)
            signalRows.append(r)
            signalValues.append(value)
        }
        let signalStack = column(signalRows, in: signal, spacing: 4)
        signalStack.setCustomSpacing(8, after: signalRows[0])

        // 2. Processing stages
        let stages = PanelView()
        let header = NSStackView(views: [Theme.heading("Processing · per game frame"), budgetLabel])
        header.orientation = .horizontal
        header.distribution = .equalSpacing
        header.translatesAutoresizingMaskIntoConstraints = false
        var stageRows: [NSView] = [header]
        for name in ["Clean-up", "Sharpen + colour", "Frame generation", "Upscaling"] {
            let label = Theme.label(name, 11.5, .regular, Theme.muted)
            label.widthAnchor.constraint(equalToConstant: 104).isActive = true
            let bar = BarView(frame: .zero)
            let value = Theme.label("–", 11.5, .medium)
            value.font = Theme.mono(11.5, .medium)
            value.alignment = .right
            value.widthAnchor.constraint(equalToConstant: 84).isActive = true
            let r = NSStackView(views: [label, bar, value])
            r.orientation = .horizontal
            r.alignment = .centerY
            r.spacing = 10
            r.translatesAutoresizingMaskIntoConstraints = false
            stageRows.append(r)
            stageBars.append(bar)
            stageValues.append(value)
        }
        let lanes = Theme.wrapping("Enhancement runs side by side with frame generation + upscaling; the busier lane must stay under budget.", 10.5, Theme.faint, width: 380)
        stageRows.append(lanes)
        let stageStack = column(stageRows, in: stages, spacing: 11)

        // 3. Frame rate and pacing
        let pacing = PanelView()
        fpsSub.maximumNumberOfLines = 2
        let fpsRow = NSStackView(views: [fpsBig, fpsSub])
        fpsRow.orientation = .horizontal
        fpsRow.alignment = .lastBaseline
        fpsRow.spacing = 8
        fpsRow.translatesAutoresizingMaskIntoConstraints = false
        graph.heightAnchor.constraint(equalToConstant: 58).isActive = true
        let left = column([Theme.heading("Frame rate and pacing"), fpsRow, graph], in: pacing, spacing: 6)
        var gaugeViews: [NSView] = []
        for (gauge, caption) in [(latencyGauge, "Latency"), (processingGauge, "Processing"), (loadGauge, "Load")] {
            let c = Theme.label(caption, 10.5, .medium, Theme.faint)
            let v = NSStackView(views: [gauge, c])
            v.orientation = .vertical
            v.alignment = .centerX
            v.spacing = 6
            gaugeViews.append(v)
        }
        let gauges = NSStackView(views: gaugeViews)
        gauges.orientation = .horizontal
        gauges.spacing = 8
        gauges.translatesAutoresizingMaskIntoConstraints = false
        pacing.addSubview(gauges)

        for panel in [signal, stages, pacing] { row.addSubview(panel) }
        let inset: CGFloat = 14
        NSLayoutConstraint.activate([
            signal.leadingAnchor.constraint(equalTo: row.leadingAnchor),
            signal.topAnchor.constraint(equalTo: row.topAnchor),
            signal.bottomAnchor.constraint(equalTo: row.bottomAnchor),
            signal.widthAnchor.constraint(equalTo: row.widthAnchor, multiplier: 0.25),
            stages.leadingAnchor.constraint(equalTo: signal.trailingAnchor, constant: 10),
            stages.topAnchor.constraint(equalTo: row.topAnchor),
            stages.bottomAnchor.constraint(equalTo: row.bottomAnchor),
            stages.widthAnchor.constraint(equalTo: row.widthAnchor, multiplier: 0.33),
            pacing.leadingAnchor.constraint(equalTo: stages.trailingAnchor, constant: 10),
            pacing.trailingAnchor.constraint(equalTo: row.trailingAnchor),
            pacing.topAnchor.constraint(equalTo: row.topAnchor),
            pacing.bottomAnchor.constraint(equalTo: row.bottomAnchor),

            signalStack.leadingAnchor.constraint(equalTo: signal.leadingAnchor, constant: inset),
            signalStack.trailingAnchor.constraint(equalTo: signal.trailingAnchor, constant: -inset),
            signalStack.topAnchor.constraint(equalTo: signal.topAnchor, constant: 12),
            stageStack.leadingAnchor.constraint(equalTo: stages.leadingAnchor, constant: inset),
            stageStack.trailingAnchor.constraint(equalTo: stages.trailingAnchor, constant: -inset),
            stageStack.topAnchor.constraint(equalTo: stages.topAnchor, constant: 12),
            left.leadingAnchor.constraint(equalTo: pacing.leadingAnchor, constant: inset),
            left.topAnchor.constraint(equalTo: pacing.topAnchor, constant: 12),
            left.trailingAnchor.constraint(equalTo: gauges.leadingAnchor, constant: -16),
            gauges.trailingAnchor.constraint(equalTo: pacing.trailingAnchor, constant: -inset),
            gauges.centerYAnchor.constraint(equalTo: pacing.centerYAnchor, constant: 6),
        ])
        return row
    }

    private func updateStats(simple: String, detailed: String, live: LiveStats) {
        guard running else { return }
        statsLabel.stringValue = detailedOverlay ? detailed : simple
        videoMessage.isHidden = true

        let res = { (w: Int, h: Int) in "\(w)×\(h)" }
        if live.outputWidth > 0, (live.outputWidth, live.outputHeight) != lastOutputSize {
            lastOutputSize = (live.outputWidth, live.outputHeight)
            sizeStreamWindow()
        }
        cardChip.stringValue = live.device
        signalChip.stringValue = String(format: "%@ · %.0f fps in · game %.0f", res(live.inputWidth, live.inputHeight), live.inputFPS, live.gameFPS)
        outputChip.stringValue = String(format: "Output %@ · %.0f fps", res(live.outputWidth, live.outputHeight), live.outputFPS)

        let values = [
            live.device,
            String(format: "%@ · %.0f fps", res(live.inputWidth, live.inputHeight), live.inputFPS)
                + (live.processWidth > 0 && live.processWidth != live.inputWidth ? " → \(live.processHeight)p" : ""),
            String(format: "%.0f fps", live.gameFPS),
            String(format: "%@ · %.0f fps", res(live.outputWidth, live.outputHeight), live.outputFPS),
            live.frameGeneration,
            live.upscaling,
            live.enhancement + (live.subtitleStrip > 0 ? " · subtitles \(live.subtitleStrip)%" : ""),
            "\(live.screenHz) Hz",
        ]
        for (label, value) in zip(signalValues, values) { label.stringValue = value }

        budgetLabel.stringValue = String(format: "Budget %.1f ms", live.budgetMs)
        let upscaleTotal = live.upscaleMs * Double(live.upscaledFramesPerGameFrame)
        let stageMs: [(Double, String?)] = [
            (live.cleanUpMs, live.cleanUpMs > 0 ? nil : "off"),
            (live.pictureMs, live.pictureMs > 0 ? nil : "off"),
            (live.generateMs, live.generateMs > 0 ? nil : "off"),
            (upscaleTotal, upscaleTotal > 0 ? String(format: "%.1f ms (%d×)", upscaleTotal, live.upscaledFramesPerGameFrame) : "off"),
        ]
        for (i, (ms, text)) in stageMs.enumerated() {
            stageBars[i].fraction = CGFloat(ms / max(live.budgetMs, 1))
            stageValues[i].stringValue = text ?? String(format: "%.1f ms", ms)
            stageValues[i].textColor = ms > 0 ? Theme.text : Theme.faint
        }

        updateCompareTags(live)
        videoView.protectedRects = live.protectedRects
        videoView.contentSize = CGSize(width: max(1, live.outputWidth), height: max(1, live.outputHeight))
        updateAudioHint(live)
        fpsBig.stringValue = String(format: "%.1f", live.outputFPS)
        let low = live.onePercentLow.map { String(format: "1%% low %.0f", $0) } ?? "1% low –"
        fpsSub.stringValue = "fps output\n\(low) · game \(Int(live.gameFPS.rounded()))"
        fpsHistory.append(live.outputFPS)
        if fpsHistory.count > graph.capacity { fpsHistory.removeFirst(fpsHistory.count - graph.capacity) }
        graph.target = Double(currentFPS == 120 ? 120 : 60)
        graph.values = fpsHistory

        latencyGauge.valueText = String(format: "%.0f ms", live.latencyMs)
        latencyGauge.fraction = CGFloat(live.latencyMs / 150)
        let lane = max(live.enhanceLaneMs, live.generateLaneMs)
        processingGauge.valueText = String(format: "%.1f ms", lane)
        processingGauge.fraction = CGFloat(lane / max(live.budgetMs, 1))
        loadGauge.valueText = String(format: "%.0f%%", live.load * 100)
        loadGauge.fraction = CGFloat(live.load)

        var steps = ["Input \(res(live.inputWidth, live.inputHeight))"]
        if live.processWidth > 0 && live.processWidth != live.inputWidth {
            steps.append("scaled to \(res(live.processWidth, live.processHeight)) for frame gen")
        }
        if live.enhancement != "Off" { steps.append("enhance \(live.enhancement)") }
        if live.generateMs > 0 { steps.append("frame gen \(live.frameGeneration.components(separatedBy: " ").first ?? "")") }
        if live.upscaleMs > 0 { steps.append("MetalFX \(res(live.outputWidth, live.outputHeight))") }
        steps.append(String(format: "%.0f fps on a %d Hz screen", live.outputFPS, live.screenHz))
        pipelineLabel.stringValue = steps.joined(separator: " → ")
    }

    private func showIdle() {
        cardChip.stringValue = cardPopup.selectedItem?.title ?? "No capture card"
        signalChip.stringValue = "Not running"
        outputChip.stringValue = "Press Start"
        for label in signalValues { label.stringValue = "–" }
        for bar in stageBars { bar.fraction = 0 }
        for label in stageValues { label.stringValue = "–"; label.textColor = Theme.faint }
        budgetLabel.stringValue = "Budget 33.3 ms"
        fpsBig.stringValue = "–"
        fpsSub.stringValue = "fps output\n1% low – · game –"
        fpsHistory.removeAll()
        graph.values = []
        for gauge in [latencyGauge, processingGauge, loadGauge] { gauge.valueText = "–"; gauge.fraction = 0 }
        pipelineLabel.stringValue = "Not running. Pick your capture card, then press Start."
        showVideoMessage("Press Start to play from your capture card")
    }

    private func showVideoMessage(_ text: String) {
        videoMessage.stringValue = text
        videoMessage.isHidden = false
    }

    private func showNotice(_ text: String) {
        noticeLabel.stringValue = "   \(text)   "
        noticeLabel.isHidden = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
            MainActor.assumeIsolated { self?.noticeLabel.isHidden = true }
        }
    }

    // MARK: Settings → engine

    /// Puts every control and menu tick in line with the saved settings.
    private func syncControls() {
        fpsPopup.selectItem(withTag: savedFPS)
        if !upscalePopup.selectItem(withTag: savedUpscale) { upscalePopup.selectItem(withTag: 0) }
        upscalePopup.isEnabled = savedFPS == 60 && !upscaleTargets.isEmpty
        enhancePopup.selectItem(withTag: savedEnhance.rawValue)
        rangePopup.selectItem(withTag: savedRange.rawValue)
        frameGenSwitch.isOn = frameGenOn
        hudSwitch.isOn = savedHUD
        hudSwitch.isEnabled = savedEnhance != .off
        subtitlePopup.selectItem(withTag: savedSubtitles.rawValue)
        subtitlePopup.isEnabled = savedEnhance != .off && savedHUD
        protectedSwitch.isOn = showProtected
        protectedSwitch.isEnabled = savedEnhance != .off && savedHUD
        videoView.showsProtected = showProtected && savedEnhance != .off && savedHUD
        overlaySwitch.isOn = showOverlay
        detailedSwitch.isOn = detailedOverlay
        statsBox.isHidden = !showOverlay

        frameGenItem?.state = frameGenOn ? .on : .off
        statsItem?.state = showOverlay ? .on : .off
        detailedStatsItem?.state = detailedOverlay ? .on : .off
        for item in rangeItems { item.state = item.tag == savedRange.rawValue ? .on : .off }
        for item in enhanceItems { item.state = item.tag == savedEnhance.rawValue ? .on : .off }
        hudItem?.state = savedHUD ? .on : .off

        // Hints
        let chipName: String? = {
            let tag = macPopup.selectedItem?.tag ?? -1
            if tag >= 0 && tag < MacCatalog.models.count { return MacCatalog.models[tag].chip }
            let detected = MacCatalog.detectedChip()
            return detected.isEmpty ? nil : detected
        }()
        var source: [String] = []
        if let index = detectedIndex { source.append("Detected: \(MacCatalog.models[index].name)") }
        source.append("Game audio is taken from the capture card and kept in sync with the picture.")
        sourceHint.stringValue = source.joined(separator: "\n\n")

        var fg: [String] = []
        if savedFPS == 120 {
            fg.append(MacCatalog.advice120(for: chipName))
            let hz = NSScreen.screens.map { $0.maximumFramesPerSecond }.max() ?? 60
            if hz < 110 { fg.append("Your screens run at \(hz) Hz, so you'll only see 60 fps. Use a 120 Hz display to see 120.") }
        } else {
            fg.append("60 fps works on every Apple silicon Mac.")
        }
        frameGenHint.stringValue = fg.joined(separator: "\n\n")

        var quality: [String] = []
        if savedFPS == 120 { quality.append("Upscaling is off at 120 fps — switch Frame Gen to 60 fps to use it.") }
        if savedEnhance.denoiseStrength > 0 {
            quality.append("Clean-up removes capture noise and shimmer. If the Mac can't keep up, it switches clean-up off and keeps sharpen + colour.")
        }
        quality.append("Recordings and screenshots include upscaling and enhancement.")
        qualityHint.stringValue = quality.joined(separator: "\n\n")
    }

    private func applyUpscaling() {
        let tag = savedUpscale
        capture.setUpscale(targetHeight: currentFPS == 60 && tag >= 1000 ? tag : 0)
    }

    private func applyEnhancement() {
        capture.setEnhancement(savedEnhance, protectHUD: savedHUD, subtitles: savedSubtitles)
        syncControls()
    }

    private func applyColourRange() {
        capture.setColourRange(savedRange)
        syncControls()
    }

    private func setFrameGeneration(_ on: Bool) {
        defaults.set(on, forKey: "frameGeneration")
        capture.setFrameGeneration(on)
        syncControls()
    }

    private func setOverlay(_ on: Bool) {
        defaults.set(on, forKey: "showOverlay")
        syncControls()
    }

    @objc private func macChanged() {
        defaults.set(macPopup.selectedItem?.tag ?? -1, forKey: "macModelIndex")
        syncControls()
    }

    @objc private func cardChanged() {
        guard let id = cardPopup.selectedItem?.representedObject as? String else { return }
        defaults.set(id, forKey: "captureDeviceID")
        if !running { cardChip.stringValue = cardPopup.selectedItem?.title ?? "" }
        if running, id != currentDeviceID, let device = CaptureController.videoDevices().first(where: { $0.uniqueID == id }) {
            start(device)
        }
    }

    @objc private func fpsChanged() {
        let fps = fpsPopup.selectedTag() == 120 ? 120 : 60
        defaults.set(fps, forKey: "targetFPS")
        if running {
            currentFPS = fps
            capture.setTargetFPS(fps)
            applyUpscaling()
        }
        syncControls()
    }

    @objc private func upscaleChanged() {
        defaults.set(upscalePopup.selectedTag(), forKey: "upscaleTag")
        if running { applyUpscaling() }
        syncControls()
    }

    @objc private func subtitleModeChanged() {
        defaults.set(subtitlePopup.selectedTag(), forKey: "subtitleMode")
        applyEnhancement()
    }

    @objc private func enhanceChanged() {
        defaults.set(enhancePopup.selectedTag(), forKey: "enhanceLevel")
        applyEnhancement()
    }

    @objc private func rangeChanged() {
        defaults.set(rangePopup.selectedTag(), forKey: "colourRange")
        applyColourRange()
    }

    @objc private func resetSettings() {
        for key in ["targetFPS", "frameGeneration", "upscaleTag", "enhanceLevel", "protectHUD", "subtitleMode", "showProtected", "colourRange",
                    "showOverlay", "detailedStats", "recordingsFolder", "screenshotsFolder",
                    "soundOn", "soundVolume", "audioSyncMs", "audioOutputUID"] {
            defaults.removeObject(forKey: key)
        }
        updateFolderLabels()
        applyAudio()
        refreshAudioOutputs()
        if running {
            currentFPS = savedFPS
            capture.setTargetFPS(currentFPS)
            capture.setFrameGeneration(frameGenOn)
            applyUpscaling()
            capture.setEnhancement(savedEnhance, protectHUD: savedHUD, subtitles: savedSubtitles)
            capture.setColourRange(savedRange)
        }
        syncControls()
        showNotice("Settings reset")
    }

    // MARK: Capture devices

    private func refreshDevices() {
        let devices = CaptureController.videoDevices()
        let current = (cardPopup.selectedItem?.representedObject as? String) ?? currentDeviceID
        cardPopup.removeAllItems()
        for device in devices {
            cardPopup.addItem(withTitle: device.localizedName)
            cardPopup.lastItem?.representedObject = device.uniqueID
        }
        // The saved choice, unless it's an iPhone/virtual camera and a real capture card is plugged in.
        let saved = current ?? defaults.string(forKey: "captureDeviceID")
        let savedDevice = devices.first(where: { $0.uniqueID == saved })
        let bestCard = devices.first(where: { CaptureController.deviceRank($0) == 0 })
        let choice: AVCaptureDevice?
        if let savedDevice, !(CaptureController.deviceRank(savedDevice) >= 2 && bestCard != nil) {
            choice = savedDevice
        } else {
            choice = bestCard ?? devices.first
        }
        if let choice, let item = cardPopup.itemArray.first(where: { ($0.representedObject as? String) == choice.uniqueID }) {
            cardPopup.select(item)
        }
        if devices.isEmpty {
            cardPopup.addItem(withTitle: "No capture card found")
            cardPopup.lastItem?.isEnabled = false
        }
        if !running { cardChip.stringValue = cardPopup.selectedItem?.title ?? "No capture card" }
    }

    @objc private func rescanDevices() { refreshDevices() }

    func menuNeedsUpdate(_ menu: NSMenu) {
        if menu === cardPopup.menu {
            refreshDevices()
            return
        }
        if menu === audioOutputPopup.menu {
            refreshAudioOutputs()
            return
        }
        guard menu === inputMenu else { return }
        menu.removeAllItems()
        let devices = CaptureController.videoDevices()
        if devices.isEmpty {
            let empty = NSMenuItem(title: "No capture devices found", action: nil, keyEquivalent: "")
            empty.isEnabled = false
            menu.addItem(empty)
            return
        }
        for device in devices {
            let item = NSMenuItem(title: device.localizedName, action: #selector(chooseDevice(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = device.uniqueID
            item.state = device.uniqueID == (currentDeviceID ?? defaults.string(forKey: "captureDeviceID")) ? .on : .off
            menu.addItem(item)
        }
    }

    @objc private func chooseDevice(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        defaults.set(id, forKey: "captureDeviceID")
        refreshDevices()
        if let item = cardPopup.itemArray.first(where: { ($0.representedObject as? String) == id }) { cardPopup.select(item) }
        if running, let device = CaptureController.videoDevices().first(where: { $0.uniqueID == id }) { start(device) }
    }

    // MARK: Start / stop

    @objc private func startStopPressed() {
        running ? stopPlaying() : startPlaying()
    }

    private func updateStartButton() {
        startButton.setLabel(running ? "Stop" : "Start")
        startButton.image = NSImage(systemSymbolName: running ? "stop.fill" : "play.fill", accessibilityDescription: nil)?
            .withSymbolConfiguration(.init(pointSize: 12, weight: .semibold))
        startButton.setFill(running ? Theme.raised : Theme.red)
        startItem?.title = running ? "Stop" : "Start"
        recordButton.isEnabled = running
        shotButton.isEnabled = running
        compareButton.isEnabled = running
        recordButton.alphaValue = running ? 1 : 0.5
        shotButton.alphaValue = running ? 1 : 0.5
        compareButton.alphaValue = running ? 1 : 0.5
    }

    private func startPlaying() {
        currentFPS = savedFPS
        capture.setTargetFPS(currentFPS)
        capture.setColourRange(savedRange)
        applyUpscaling()
        capture.setEnhancement(savedEnhance, protectHUD: savedHUD, subtitles: savedSubtitles)
        capture.setFrameGeneration(frameGenOn)
        applyAudio()
        running = true
        updateStartButton()
        fpsHistory.removeAll()
        showVideoMessage("Starting…")
        signalChip.stringValue = "Connecting…"
        updateDisplayRefresh()
        let deviceID = cardPopup.selectedItem?.representedObject as? String

        AVCaptureDevice.requestAccess(for: .video) { granted in
            AVCaptureDevice.requestAccess(for: .audio) { _ in
                Task { @MainActor in
                    guard self.running else { return }
                    guard granted else {
                        self.showVideoMessage("Camera access is off. Turn on Smooth60 in System Settings → Privacy & Security → Camera, then press Start again.")
                        self.running = false
                        self.updateStartButton()
                        return
                    }
                    let devices = CaptureController.videoDevices()
                    if let device = devices.first(where: { $0.uniqueID == deviceID }) ?? devices.first {
                        self.start(device)
                    } else {
                        self.showVideoMessage("No capture card found. Plug it in over USB, then press Rescan in the Source tab.")
                        self.running = false
                        self.updateStartButton()
                    }
                }
            }
        }
    }

    private func start(_ device: AVCaptureDevice) {
        showVideoMessage("Connecting to \(device.localizedName)…")
        currentDeviceID = device.uniqueID
        cardChip.stringValue = device.localizedName
        let capture = self.capture!
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try capture.start(device: device)
            } catch {
                let message = "Couldn't open \(device.localizedName): \(error.localizedDescription)"
                Task { @MainActor in self.showVideoMessage(message) }
            }
        }
    }

    private func stopPlaying() {
        if capture.presenterForUI.recorder.isRecording { toggleRecording() }
        if videoView.comparing { toggleCompare() }
        running = false
        capture.stop()
        videoView.displayLayer.sampleBufferRenderer.flush(removingDisplayedImage: true, completionHandler: nil)
        updateStartButton()
        showIdle()
    }

    // MARK: Recording and screenshots

    @objc private func toggleRecording() {
        guard running else { return }
        let recorder = capture.presenterForUI.recorder
        if recorder.isRecording {
            recordTimer?.invalidate()
            recordTimer = nil
            recBadge.isHidden = true
            recordItem.title = "Start Recording"
            recordButton.setLabel("Record")
            recordButton.setFill(Theme.raised)
            recordButton.contentTintColor = Theme.red
            recorder.stop { url, error in
                Task { @MainActor in
                    if let url {
                        self.showNotice("Recording saved: \(AppDelegate.shortPath(url))")
                    } else {
                        self.showNotice("Recording failed: \(error ?? "unknown error")")
                    }
                }
            }
        } else {
            _ = recorder.start(fps: currentFPS)
            recordingStarted = Date()
            recordItem.title = "Stop Recording"
            recordButton.setFill(Theme.red)
            recordButton.contentTintColor = .white
            updateRecBadge()
            recBadge.isHidden = false
            recordTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.updateRecBadge() }
            }
        }
    }

    private func updateRecBadge() {
        let seconds = Int(Date().timeIntervalSince(recordingStarted ?? Date()))
        let time = String(format: "%d:%02d", seconds / 60, seconds % 60)
        recBadge.stringValue = "  ● REC  \(time)  "
        recordButton.setLabel("Stop · \(time)")
    }

    @objc private func takeScreenshot() {
        guard running, let frame = capture.presenterForUI.latestFrame else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            let url = Screenshotter.save(frame)
            Task { @MainActor in
                self.showNotice(url.map { "Screenshot saved: \(AppDelegate.shortPath($0))" } ?? "Couldn't save the screenshot")
            }
        }
    }

    private func openFolder(_ kind: String) {
        let url = Recorder.saveFolder(kind)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        NSWorkspace.shared.open(url)
    }
    @objc private func openRecordings() { openFolder("Recordings") }
    @objc private func openScreenshots() { openFolder("Screenshots") }

    /// "~/Movies/Smooth60/clip.mp4" → "Movies › Smooth60 › clip.mp4"
    static func shortPath(_ url: URL) -> String {
        var path = url.path
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if path.hasPrefix(home + "/") { path = String(path.dropFirst(home.count + 1)) }
        else if path.hasPrefix("/Volumes/") { path = String(path.dropFirst("/Volumes/".count)) }
        return path.split(separator: "/").joined(separator: " › ")
    }

    private func updateFolderLabels() {
        let recordings = Recorder.saveFolder("Recordings")
        let shots = Recorder.saveFolder("Screenshots")
        recordingsPathLabel.stringValue = "\(AppDelegate.shortPath(recordings))\nF8 or ⌘R"
        screenshotsPathLabel.stringValue = "\(AppDelegate.shortPath(shots))\nF7 or ⌘S"
        recordingsPathLabel.toolTip = recordings.path
        screenshotsPathLabel.toolTip = shots.path
    }

    private func chooseFolder(kind: String, key: String) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Use This Folder"
        panel.message = "Choose where Smooth60 saves \(kind.lowercased())"
        panel.directoryURL = Recorder.saveFolder(kind)
        panel.beginSheetModal(for: window) { response in
            MainActor.assumeIsolated {
                guard response == .OK, let url = panel.url else { return }
                self.defaults.set(url.path, forKey: key)
                self.updateFolderLabels()
                self.showNotice("\(kind) will be saved to \(AppDelegate.shortPath(url))")
            }
        }
    }
    @objc private func changeRecordingsFolder() { chooseFolder(kind: "Recordings", key: "recordingsFolder") }
    @objc private func changeScreenshotsFolder() { chooseFolder(kind: "Screenshots", key: "screenshotsFolder") }

    // MARK: Sound

    /// Applies the saved sound settings to the player.
    private func applyAudio() {
        clock.setAudio(muted: !soundOn, volume: Float(savedVolume), outputUID: savedAudioOutput)
        capture.setAudioSync(ms: savedAudioSync)
        soundButton.setLabel(soundOn ? "Sound" : "Muted")
        soundButton.image = NSImage(systemSymbolName: soundOn ? "speaker.wave.2.fill" : "speaker.slash.fill",
                                    accessibilityDescription: nil)?.withSymbolConfiguration(.init(pointSize: 12, weight: .semibold))
        soundButton.setFill(soundOn ? Theme.red : Theme.raised)
        soundSwitch.isOn = soundOn
        soundItem?.state = soundOn ? .on : .off
        volumeSlider.doubleValue = savedVolume
        syncSlider.doubleValue = savedAudioSync
        syncValue.stringValue = String(format: "%+.0f ms", savedAudioSync)
    }

    private func setSound(_ on: Bool) {
        defaults.set(on, forKey: "soundOn")
        applyAudio()
    }
    @objc private func toggleSound() { setSound(!soundOn) }

    private func refreshAudioOutputs() {
        let devices = AudioOutputs.all()
        audioOutputPopup.removeAllItems()
        let systemName = AudioOutputs.defaultName().map { "System default (\($0))" } ?? "System default"
        audioOutputPopup.addItem(withTitle: systemName)
        audioOutputPopup.lastItem?.representedObject = nil
        for device in devices {
            audioOutputPopup.addItem(withTitle: device.name)
            audioOutputPopup.lastItem?.representedObject = device.uid
        }
        if let uid = savedAudioOutput,
           let item = audioOutputPopup.itemArray.first(where: { ($0.representedObject as? String) == uid }) {
            audioOutputPopup.select(item)
        } else {
            audioOutputPopup.selectItem(at: 0)
        }
    }

    @objc private func audioOutputChanged() {
        if let uid = audioOutputPopup.selectedItem?.representedObject as? String {
            defaults.set(uid, forKey: "audioOutputUID")
        } else {
            defaults.removeObject(forKey: "audioOutputUID")
        }
        applyAudio()
        if !soundOn { setSound(true) }
        showNotice("Sound now plays on \(audioOutputPopup.selectedItem?.title ?? "the default output")")
    }

    @objc private func volumeChanged() {
        defaults.set(volumeSlider.doubleValue, forKey: "soundVolume")
        applyAudio()
    }

    @objc private func audioSyncChanged() {
        defaults.set((syncSlider.doubleValue / 10).rounded() * 10, forKey: "audioSyncMs")
        applyAudio()
    }

    @objc private func resetAudioSync() {
        defaults.removeObject(forKey: "audioSyncMs")
        applyAudio()
    }

    private func updateAudioHint(_ live: LiveStats) {
        var lines: [String] = []
        if live.audio == "none" || live.audio.isEmpty {
            lines.append("No sound device found for this capture card — check it's plugged in over USB and that Microphone access is allowed for Smooth60 (System Settings → Privacy & Security → Microphone).")
        } else if live.audioPacketsPerSecond < 1 {
            lines.append("\(live.audio) is connected but sending no sound. Check the PS5's audio output (Settings → Sound → Audio Output → HDMI device).")
        } else {
            lines.append(String(format: "Receiving game sound from %@ · %@.", live.audio, soundOn ? "playing" : "muted"))
        }
        if live.audioPlayback.hasPrefix("failed") || live.audioPlayback.hasPrefix("recovering") {
            lines.append("Sound player problem (\(live.audioPlayback)) — Smooth60 restarts the player automatically. If it keeps happening, pick another output above.")
        }
        lines.append(String(format: "Sound is delayed %.0f ms to match the picture, so they stay in sync. If Streamlabs also plays the card's sound, turn its monitoring off so you don't hear it twice.", live.latencyMs))
        audioHint.stringValue = lines.joined(separator: "\n\n")
    }

    // MARK: Before / After

    /// Left of the divider: the capture card's picture exactly as it arrives (game fps, no processing).
    /// Right: Smooth60's output. Both play on the same clock, so they line up frame for frame.
    @objc private func toggleCompare() {
        let on = !videoView.comparing
        guard !on || running else { return }
        let before = videoView.beforeLayer
        if on {
            clock.synchronizer.addRenderer(before)
            capture.presenterForUI.setBefore(before.sampleBufferRenderer)
            videoView.split = 0.5
            showNotice("Before / After: drag the divider to compare. Left = original, right = Smooth60")
        } else {
            capture.presenterForUI.setBefore(nil)
            clock.synchronizer.removeRenderer(before, at: clock.synchronizer.currentTime(), completionHandler: nil)
            before.sampleBufferRenderer.flush(removingDisplayedImage: true, completionHandler: nil)
        }
        videoView.comparing = on
        compareButton.setFill(on ? Theme.red : Theme.raised)
        compareItem?.state = on ? .on : .off
    }

    private func updateCompareTags(_ live: LiveStats) {
        guard videoView.comparing else { return }
        videoView.beforeTag.stringValue = String(format: "BEFORE · %.0f fps · %d×%d", live.gameFPS, live.inputWidth, live.inputHeight)
        var after = String(format: "AFTER · %.0f fps · %d×%d", live.outputFPS, live.outputWidth, live.outputHeight)
        if live.enhancement != "Off" { after += " · \(live.enhancement)" }
        videoView.afterTag.stringValue = after
        videoView.needsLayout = true
    }

    // MARK: Stream output window

    /// Opens or closes a borderless window showing only the processed video, sized to the
    /// output resolution in real pixels, for Window Capture in Streamlabs / OBS.
    private func setStreamOutput(_ on: Bool) {
        if on {
            guard streamWindow == nil else { return }
            let view = VideoView(frame: NSRect(x: 0, y: 0, width: 960, height: 540))
            view.autoresizingMask = [.width, .height]
            let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 540), styleMask: [.borderless],
                             backing: .buffered, defer: false)
            w.title = "Smooth60 Stream Output"
            w.contentView = view
            w.backgroundColor = .black
            w.isMovableByWindowBackground = true
            w.isReleasedWhenClosed = false
            w.hasShadow = false
            w.collectionBehavior = [.managed, .ignoresCycle, .fullScreenAuxiliary]
            // Always behind every other window (it's as big as the screen); window capture still sees it.
            w.level = NSWindow.Level(rawValue: NSWindow.Level.normal.rawValue - 1)
            streamWindow = w
            streamView = view
            clock.synchronizer.addRenderer(view.displayLayer)
            capture.presenterForUI.addMirror(view.displayLayer.sampleBufferRenderer)
            if let screen = window.screen ?? NSScreen.main {
                w.setFrameOrigin(NSPoint(x: screen.frame.minX, y: screen.frame.maxY - w.frame.height))
            }
            sizeStreamWindow()
            w.order(.below, relativeTo: window.windowNumber)   // stays behind; window capture still sees it
            showNotice("Stream output is on — in Streamlabs / OBS add Window Capture → “Smooth60 Stream Output”")
        } else {
            guard let w = streamWindow, let view = streamView else { return }
            capture.presenterForUI.removeMirror(view.displayLayer.sampleBufferRenderer)
            clock.synchronizer.removeRenderer(view.displayLayer, at: clock.synchronizer.currentTime(), completionHandler: nil)
            w.orderOut(nil)
            streamWindow = nil
            streamView = nil
        }
        streamSwitch.isOn = on
        streamItem?.state = on ? .on : .off
        defaults.set(on, forKey: "streamOutput")   // remembered for next launch
    }

    /// The output size in real pixels (so 1080p stays 1080p in the stream), shrunk to fit the
    /// screen when the output is bigger (e.g. 4K on a 1080p monitor) so capture sees all of it.
    private func sizeStreamWindow() {
        guard let w = streamWindow, let screen = w.screen ?? window.screen ?? NSScreen.main else { return }
        let scale = screen.backingScaleFactor
        var size = NSSize(width: CGFloat(lastOutputSize.0) / scale, height: CGFloat(lastOutputSize.1) / scale)
        let fit = min(1, screen.frame.width / size.width, screen.frame.height / size.height)
        size = NSSize(width: (size.width * fit).rounded(), height: (size.height * fit).rounded())
        let top = min(w.frame.maxY, screen.frame.maxY)
        let left = max(screen.frame.minX, min(w.frame.minX, screen.frame.maxX - size.width))
        w.setFrame(NSRect(x: left, y: top - size.height, width: size.width, height: size.height), display: true)
    }

    @objc private func menuToggleStream() { setStreamOutput(streamWindow == nil) }

    // MARK: Full screen = video only

    @objc private func goFullScreen() { window.toggleFullScreen(nil) }

    func windowWillEnterFullScreen(_ notification: Notification) { setPanelsHidden(true) }
    func windowWillExitFullScreen(_ notification: Notification) { setPanelsHidden(false) }

    private func setPanelsHidden(_ hidden: Bool) {
        for view in [topBar!, sidebar!, bottomRow!, titleStrip!] as [NSView] { view.isHidden = hidden }
        if hidden {
            NSLayoutConstraint.deactivate(normalConstraints)
            NSLayoutConstraint.activate(fullScreenConstraints)
            videoPanel.layer?.cornerRadius = 0
            videoPanel.layer?.borderWidth = 0
        } else {
            NSLayoutConstraint.deactivate(fullScreenConstraints)
            NSLayoutConstraint.activate(normalConstraints)
            videoPanel.layer?.cornerRadius = 10
            videoPanel.layer?.borderWidth = 1
        }
    }

    // MARK: Menus

    private func buildMenus() {
        let main = NSMenu()

        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About Smooth60", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit Smooth60", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)

        let viewItem = NSMenuItem()
        let viewMenu = NSMenu(title: "View")
        startItem = NSMenuItem(title: "Start", action: #selector(startStopPressed), keyEquivalent: "\r")
        startItem.target = self
        viewMenu.addItem(startItem)
        viewMenu.addItem(.separator())
        frameGenItem = NSMenuItem(title: "Frame Generation", action: #selector(menuToggleFrameGeneration), keyEquivalent: "g")
        frameGenItem.target = self
        viewMenu.addItem(frameGenItem)
        statsItem = NSMenuItem(title: "Stats Box Over Video", action: #selector(menuToggleOverlay), keyEquivalent: "i")
        statsItem.target = self
        viewMenu.addItem(statsItem)
        detailedStatsItem = NSMenuItem(title: "Detailed Stats", action: #selector(menuToggleDetailed), keyEquivalent: "i")
        detailedStatsItem.keyEquivalentModifierMask = [.command, .option]
        detailedStatsItem.target = self
        viewMenu.addItem(detailedStatsItem)
        let rangeMenu = NSMenu(title: "Colour Range")
        for range in [ColourRange.auto, .limited, .full] {
            let item = NSMenuItem(title: range.title, action: #selector(menuChooseRange(_:)), keyEquivalent: "")
            item.target = self
            item.tag = range.rawValue
            rangeMenu.addItem(item)
            rangeItems.append(item)
        }
        let rangeParent = NSMenuItem(title: "Colour Range", action: nil, keyEquivalent: "")
        rangeParent.submenu = rangeMenu
        viewMenu.addItem(rangeParent)
        let enhanceMenu = NSMenu(title: "Enhancement")
        for level in [EnhanceLevel.off, .light, .medium, .strong] {
            let item = NSMenuItem(title: level.name, action: #selector(menuChooseEnhancement(_:)), keyEquivalent: "\(level.rawValue)")
            item.keyEquivalentModifierMask = [.command, .option]
            item.target = self
            item.tag = level.rawValue
            enhanceMenu.addItem(item)
            enhanceItems.append(item)
        }
        enhanceMenu.addItem(.separator())
        hudItem = NSMenuItem(title: "Protect HUD & Subtitles", action: #selector(menuToggleHUD), keyEquivalent: "")
        hudItem.target = self
        enhanceMenu.addItem(hudItem)
        let enhanceParent = NSMenuItem(title: "Enhancement", action: nil, keyEquivalent: "")
        enhanceParent.submenu = enhanceMenu
        viewMenu.addItem(enhanceParent)
        soundItem = NSMenuItem(title: "Game Sound", action: #selector(toggleSound), keyEquivalent: "m")
        soundItem.keyEquivalentModifierMask = [.command, .shift]
        soundItem.target = self
        viewMenu.addItem(soundItem)
        compareItem = NSMenuItem(title: "Before / After", action: #selector(toggleCompare), keyEquivalent: "b")
        compareItem.target = self
        viewMenu.addItem(compareItem)
        streamItem = NSMenuItem(title: "Stream Output Window", action: #selector(menuToggleStream), keyEquivalent: "o")
        streamItem.keyEquivalentModifierMask = [.command, .shift]
        streamItem.target = self
        viewMenu.addItem(streamItem)
        viewMenu.addItem(.separator())
        let fullScreen = NSMenuItem(title: "Enter Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        fullScreen.keyEquivalentModifierMask = [.control, .command]
        viewMenu.addItem(fullScreen)
        viewItem.submenu = viewMenu
        main.addItem(viewItem)

        let captureItem = NSMenuItem()
        let captureMenu = NSMenu(title: "Capture")
        recordItem = NSMenuItem(title: "Start Recording", action: #selector(toggleRecording), keyEquivalent: "r")
        recordItem.target = self
        captureMenu.addItem(recordItem)
        let shotItem = NSMenuItem(title: "Take Screenshot", action: #selector(takeScreenshot), keyEquivalent: "s")
        shotItem.target = self
        captureMenu.addItem(shotItem)
        captureMenu.addItem(.separator())
        let tip = NSMenuItem(title: "Tip: F8 records, F7 takes a screenshot", action: nil, keyEquivalent: "")
        tip.isEnabled = false
        captureMenu.addItem(tip)
        captureItem.submenu = captureMenu
        main.addItem(captureItem)

        let inputItem = NSMenuItem()
        inputMenu.delegate = self
        inputItem.submenu = inputMenu
        main.addItem(inputItem)

        let windowItem = NSMenuItem()
        let windowMenu = NSMenu(title: "Window")
        windowMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windowItem.submenu = windowMenu
        main.addItem(windowItem)

        NSApp.mainMenu = main
    }

    @objc private func menuToggleFrameGeneration() { setFrameGeneration(!frameGenOn) }
    @objc private func menuToggleOverlay() { setOverlay(!showOverlay) }
    @objc private func menuToggleDetailed() {
        defaults.set(!detailedOverlay, forKey: "detailedStats")
        syncControls()
    }
    @objc private func menuChooseRange(_ sender: NSMenuItem) {
        defaults.set(sender.tag, forKey: "colourRange")
        applyColourRange()
    }
    @objc private func menuChooseEnhancement(_ sender: NSMenuItem) {
        defaults.set(sender.tag, forKey: "enhanceLevel")
        applyEnhancement()
    }
    @objc private func menuToggleHUD() {
        defaults.set(!savedHUD, forKey: "protectHUD")
        applyEnhancement()
    }
}

@main
struct Smooth60App {
    @MainActor
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }
}
