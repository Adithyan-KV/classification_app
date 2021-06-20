import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';

import 'dart:async';

import 'package:camera/camera.dart';

Future<void> main() async {
  // Initialize plugins for camera
  WidgetsFlutterBinding.ensureInitialized();

  // List all available cameras
  final cameras = await availableCameras();

  // Select the first available camera
  final firstCamera = cameras.first;

  // Runs the App
  runApp(
    MaterialApp(
      theme: ThemeData(primarySwatch: Colors.blue),
      home: TakePictureScreen(
        // Pass the appropriate camera to the TakePictureScreen widget.
        camera: firstCamera,
      ),
    ),
  );
}

// A screen that allows users to take a picture using a given camera.
class TakePictureScreen extends StatefulWidget {
  final CameraDescription camera;

  const TakePictureScreen({
    Key? key,
    required this.camera,
  }) : super(key: key);

  @override
  TakePictureScreenState createState() => TakePictureScreenState();
}

class TakePictureScreenState extends State<TakePictureScreen> {
  // Controller for handling camera events
  late CameraController _controller;
  bool _cameraInitialized = false;

  //Variables for safety checks
  bool available = true;
  bool _modelLoaded = false;
  bool readyToPredict = false;

  //Instantiate new Tensorflow model
  final model = new TensorflowModel();
  late List<dynamic> currentPredictionList;

  void _initializeCamera() async {
    _controller = CameraController(
      widget.camera,
      // Use 1080p resolution.
      ResolutionPreset.veryHigh,
    );

    // Initialize camera controller
    _controller.initialize().then((_) async {
      // Start stream that other widgets can listen into
      await _controller.startImageStream((image) {
        setState(() {
          _cameraInitialized = true;
        });
        if (available) {
          // set delay
          available = false;
          //Use model to predict what is on screen
          makePrediction(image);
        }
      });
    });
  }

  void makePrediction(CameraImage image) async {
    // check if ML model has properly loaded
    if (!_modelLoaded) {
      model.loadModel();
      // Stream the predictions so that UI widgets can display them
      model.predictionStream.listen((predictions) {
        //List of predictions of what is presently on the screen
        currentPredictionList = predictions;
        if (predictions != []) {
          readyToPredict = true;
        }
      });
      _modelLoaded = true;
      //If the model is ready start making predictions
    } else {
      model.runModel(image);
    }
    // Poll every one second into the image stream to make predictions
    await Future.delayed(Duration(seconds: 1));
    available = true;
  }

  @override
  void initState() {
    //Initialize the camera
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    // Dispose of the controller when the widget is disposed.
    _controller.stopImageStream();
    _controller.dispose();
    super.dispose();
  }

  //____________________________________________________
  //WIDGET THAT HANDLES MOST OF THE UI
  //____________________________________________________
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Image Classifier')),
      // Wait till controller has initialized and then show preview
      body: Stack(
        children: <Widget>[
          //camera preview screen
          (_cameraInitialized)
              ? CameraPreview(_controller)
              : CircularProgressIndicator(),
          //Predictions screen
          Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: <Widget>[
              Container(
                padding: EdgeInsets.all(20),
                color: Colors.blue,
                height: 80,
                width: MediaQuery.of(context).size.width,
                child: Text(
                  "${currentPredictionList[0]['label']}\nConfidence:${currentPredictionList[0]['confidence'].toString()}",
                  textAlign: TextAlign.left,
                  style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w500,
                      fontSize: 16),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          )
        ],
      ),
    );
  }
}

class TensorflowModel {
  bool _modelLoaded = false;

  // Set up a predictions stream
  StreamController<List<dynamic>> _predictionStreamController =
      new StreamController();
  Stream get predictionStream => this._predictionStreamController.stream;

  // Load in the model
  Future<void> loadModel() async {
    try {
      this._predictionStreamController.add([]);
      await Tflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
      );

      //________________________________
      //FILES FOR ALTERNATE MODEL
      //________________________________
      // model: "assets/model_vgg.tflite",
      // labels: "assets/labels_cal.txt");
      _modelLoaded = true;
    } catch (e) {
      print('error loading model');
      print(e);
    }
  }

  Future<void> runModel(CameraImage image) async {
    // Start Predictions
    if (_modelLoaded) {
      // Get top 3 predictions and stream them
      List<dynamic>? predictions = await Tflite.runModelOnFrame(
        bytesList: image.planes.map((plane) {
          return plane.bytes;
        }).toList(), // required
        imageHeight: image.height,
        imageWidth: image.width,
        numResults: 3,
      );
      if (predictions!.isNotEmpty) {
        print(predictions[0].toString());
        if (this._predictionStreamController.isClosed) {
          this._predictionStreamController = StreamController();
        }
        this._predictionStreamController.add(predictions);
      } else {
        print('Empty Predictions');
      }
    } else {
      print('model not loaded');
    }
  }

  void dispose() {
    _predictionStreamController.close();
    Tflite.close();
  }
}
