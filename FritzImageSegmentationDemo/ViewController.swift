import UIKit
import AVFoundation
import ARKit
import Fritz
import CoreFoundation

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {

    @IBOutlet weak var sceneView: ARSCNView!

    private lazy var visionModel = FritzVisionPeopleSegmentationModelFast()
//    private lazy var visionModel = FritzVisionSegmentationModel(model: DeepLabV3(), name: "DeepLabV3", classes: OcclusionClass.allClasses)
//    private lazy var visionModel = FritzVisionLivingRoomSegmentationModelFast()

    var planes = [ARPlaneAnchor: SCNNode]()
    var planeColor = UIColor.init(hue: 0.5, saturation: 0.5, brightness: 0.5, alpha: 0.5)

    var maskNode : SCNNode!
    var maskMaterial : SCNMaterial!

    var currentBuffer: CVPixelBuffer?
    var currentDepthMask: CVPixelBuffer?
    
    // COREML
    var visionRequests = [VNRequest]()
    let visionQueue = DispatchQueue(label: "com.vision.ARML.visionqueue")

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set the view's delegate
        sceneView.delegate = self
        sceneView.session.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        // Create a new scene
        let scene = SCNScene(named: "art.scnassets/ship.scn")!
        
        // Set the scene to the view
        sceneView.scene = scene
        
                
        addTapGestureToSceneView()
        bilbordCreate()
        setupUIViews()
        
        // Set up Vision Model
//        guard let depthModel = try? VNCoreMLModel(for: FCRN().model) else {
//            fatalError("Could not load depth model.")
//        }
        guard let segmentationModel = try? VNCoreMLModel(for: DeepLabV3().model) else {
            fatalError("Could not load segmentation model.")
        }
        print("Finished")

//        // Set up Vision-CoreML Request
//        let depthRequest = VNCoreMLRequest(model: depthModel, completionHandler: depthCompleteHandler)
//        depthRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        let segmentationRequest = VNCoreMLRequest(model: segmentationModel, completionHandler: segmentationCompleteHandler)
        segmentationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        self.visionRequests = [segmentationRequest]
    }
    
    func setupUIViews() {
        let button = UIButton(frame: CGRect(x: 10, y: 10, width: 80, height: 30))
        button.backgroundColor = .gray
        button.titleLabel?.font =  UIFont.boldSystemFont(ofSize: 10)
        button.setTitle("Show mask", for: .normal)
        button.addTarget(self, action: #selector(buttonAction), for: .touchUpInside)
        
        self.view.addSubview(button)
    }
    
    @objc func buttonAction(sender: UIButton!) {
        if (maskMaterial.colorBufferWriteMask == .all) {
            maskMaterial.colorBufferWriteMask = .alpha
            sender.setTitle("Show mask", for: .normal)
            return
        }
        maskMaterial.colorBufferWriteMask = .all
        sender.setTitle("Hide mask", for: .normal)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = .horizontal
        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard currentBuffer == nil, case .normal = frame.camera.trackingState else {
            return
        }
        currentBuffer = frame.capturedImage

        startDetection()
    }
    
    func bilbordCreate() {
        maskMaterial = SCNMaterial()
        maskMaterial.diffuse.contents = UIColor.white
        maskMaterial.colorBufferWriteMask = .alpha
        
        let rectangle = SCNPlane(width: 0.0326, height: 0.058)
        rectangle.materials = [maskMaterial]
        
        maskNode = SCNNode(geometry: rectangle)
        maskNode?.eulerAngles = SCNVector3Make(0, 0, 0)
        maskNode?.position = SCNVector3Make(0, 0, -0.05)
        maskNode.renderingOrder = -1
        
        sceneView.pointOfView?.presentation.addChildNode(maskNode!)
    }
    
    
    //TODO: posible optimization
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(inputImage, from: inputImage.extent) {
            return cgImage
        }
        return nil
    }
    
    private func startDetection() {
        // To avoid force unwrap in VNImageRequestHandler
        guard let buffer = currentBuffer else { return }
        
        // Depth model
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .right)
        
        // Fritz AI
        let fritzImage = FritzVisionImage(imageBuffer: buffer)
        fritzImage.metadata = FritzVisionImageMetadata()
        let options = FritzVisionSegmentationModelOptions()
        options.imageCropAndScaleOption = .scaleFit
        
        //Run in background thread
        visionQueue.async {
//            self.CoreMLRequest(image: fritzImage, options: options)
            
            do {
                try imageRequestHandler.perform(self.visionRequests)
            } catch {
                print(error)
            }
        }
    }
    
    func depthCompleteHandler(request: VNRequest, error: Error?) {
        // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        guard let observations = request.results else {
            print("No results")
            return
        }
        print(observations)
    }
    
    func segmentationCompleteHandler(request: VNRequest, error: Error?) {
        // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        print("Hey")
        
        if let observations = request.results as? [VNCoreMLFeatureValueObservation] {
            if let segmentationmap = observations.first?.featureValue {
                print("Yo")
                let array = segmentationmap.multiArrayValue!
                print(array.shape)
//                let timer = ParkBenchTimer()
                let CGimage = array.cgImage(min: 0, max: 1)
//                print("The task took \(timer.stop()) seconds.")
                                
                DispatchQueue.main.async {
                    self.maskMaterial.diffuse.contents = CGimage
                }
                self.ReleaseBuffer()
            }
        }
    }
    
    
    
    private func CoreMLRequest(image : FritzVisionImage, options : FritzVisionSegmentationModelOptions) {
        
        guard let result = try? self.visionModel.predict(image, options: options) else {
            self.ReleaseBuffer()
            return
        }
        
        let timer = ParkBenchTimer()
        let maskImage = result.buildSingleClassMask(
          forClass: FritzVisionPeopleClass.person,
          clippingScoresAbove: 0.7,
          zeroingScoresBelow: 0.3)
        
//        let maskImage = result.buildSingleClassMask(
//          forClass: OcclusionClass.object,
//          clippingScoresAbove: 0.7,
//          zeroingScoresBelow: 0.3)
      
//        let maskImage = result.buildSingleClassMask(
//          forClass: FritzVisionLivingRoomClass.chair,
//            clippingScoresAbove: 0.6,
//            zeroingScoresBelow: 0.3)
//        let maskImage = result.buildMultiClassMask(withMinimumAcceptedScore: 0.7)
        
        guard let CIRef = maskImage?.ciImage else {
            self.ReleaseBuffer()
            return
        }
        guard let maskCGImage: CGImage = self.convertCIImageToCGImage(inputImage: CIRef) else {
            self.ReleaseBuffer()
            return
        }
        
//        guard let maskCGImage: CGImage = result.predictionResult.cgImage() else {
//            self.ReleaseBuffer()
//            return
//        }
        
        print("The task took \(timer.stop()) seconds.")
        
        DispatchQueue.main.async {
//            print(maskImage!.pixelBuffer()!.pixelValues())
            self.maskMaterial.diffuse.contents = maskCGImage
        }
        
        self.ReleaseBuffer()
    }
    
    private func ReleaseBuffer() {
        // The resulting image (mask) is available as observation.pixelBuffer
        // Release currentBuffer when finished to allow processing next frame
        self.currentBuffer = nil
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard let planeAnchor = anchor as? ARPlaneAnchor else { return }
        
        let width = CGFloat(planeAnchor.extent.x)
        let height = CGFloat(planeAnchor.extent.z)
        let plane = SCNPlane(width: width, height: height)
        
        plane.materials.first?.diffuse.contents = planeColor
        
        let planeNode = SCNNode(geometry: plane)
        
        let x = CGFloat(planeAnchor.center.x)
        let y = CGFloat(planeAnchor.center.y)
        let z = CGFloat(planeAnchor.center.z)
        planeNode.position = SCNVector3(x,y,z)
        planeNode.eulerAngles.x = -.pi / 2
        
        node.addChildNode(planeNode)
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let planeAnchor = anchor as?  ARPlaneAnchor,
            let planeNode = node.childNodes.first,
            let plane = planeNode.geometry as? SCNPlane
            else { return }
        
        let width = CGFloat(planeAnchor.extent.x)
        let height = CGFloat(planeAnchor.extent.z)
        plane.width = width
        plane.height = height
        
        let x = CGFloat(planeAnchor.center.x)
        let y = CGFloat(planeAnchor.center.y)
        let z = CGFloat(planeAnchor.center.z)
        planeNode.position = SCNVector3(x, y, z)
    }
    
    func addTapGestureToSceneView() {
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(ViewController.addShipToSceneView(withGestureRecognizer:)))
        sceneView.addGestureRecognizer(tapGestureRecognizer)
    }
    
    @objc func addShipToSceneView(withGestureRecognizer recognizer: UIGestureRecognizer) {
        let tapLocation = recognizer.location(in: sceneView)
        let hitTestResults = sceneView.hitTest(tapLocation, types: .existingPlaneUsingExtent)
        
        guard let hitTestResult = hitTestResults.first else { return }
        
        let boxGeometry = SCNBox(width: 0.1, height: 0.1, length: 0.1, chamferRadius: 0)
        
        let translation = SCNVector3(hitTestResult.worldTransform.columns.3.x, hitTestResult.worldTransform.columns.3.y + Float(boxGeometry.height * 0.5), hitTestResult.worldTransform.columns.3.z)
        let x = translation.x
        let y = translation.y
        let z = translation.z
        
        let cube = SCNNode(geometry: boxGeometry)
        
        cube.position = SCNVector3(x,y,z)
        sceneView.scene.rootNode.addChildNode(cube)
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
}
