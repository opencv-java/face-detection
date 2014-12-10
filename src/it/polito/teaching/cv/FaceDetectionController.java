package it.polito.teaching.cv;

import java.io.ByteArrayInputStream;
import java.util.Timer;
import java.util.TimerTask;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import javafx.application.Platform;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the face detection/tracking.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @since 2014-01-10
 * 
 */
public class FaceDetectionController
{
	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkboxes for enabling/disabling a classifier
	@FXML
	private CheckBox haarClassifier;
	@FXML
	private CheckBox lbpClassifier;
	
	// a timer for acquiring the video stream
	private Timer timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture;
	// a flag to change the button behavior
	private boolean cameraActive;
	
	// face cascade classifier
	private CascadeClassifier faceCascade;
	private int absoluteFaceSize;
	
	/**
	 * Init the controller, at start time
	 */
	protected void init()
	{
		this.capture = new VideoCapture();
		this.faceCascade = new CascadeClassifier();
		this.absoluteFaceSize = 0;
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera()
	{
		// bind an image property with the original frame container
		final ObjectProperty<Image> imageProp = new SimpleObjectProperty<>();
		this.originalFrame.imageProperty().bind(imageProp);
		
		// set a fixed width for the frame
		originalFrame.setFitWidth(380);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);
		
		if (!this.cameraActive)
		{
			// disable setting checkboxes
			this.haarClassifier.setDisable(true);
			this.lbpClassifier.setDisable(true);
			
			// start the video capture
			this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				TimerTask frameGrabber = new TimerTask() {
					@Override
					public void run()
					{
						// update the image property => update the frame
						// shown in the UI
						final Image frame = grabFrame();
						Platform.runLater(new Runnable() {
							
							@Override
							public void run()
							{
								// show the original frames
								imageProp.set(frame);
							}
						});
					}
				};
				this.timer = new Timer();
				this.timer.schedule(frameGrabber, 0, 33);
				
				// update the button content
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");
			// enable classifiers checkboxes
			this.haarClassifier.setDisable(false);
			this.lbpClassifier.setDisable(false);
			
			// stop the timer
			if (this.timer != null)
			{
				this.timer.cancel();
				this.timer = null;
			}
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Image grabFrame()
	{
		// init everything
		Image imageToShow = null;
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// face detection
					this.detectAndDisplay(frame);
					
					// convert the Mat object (OpenCV) to Image (JavaFX)
					imageToShow = mat2Image(frame);
				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.println("ERROR: " + e);
			}
		}
		
		return imageToShow;
	}
	
	/**
	 * Method for face detection and tracking
	 * 
	 * @param frame
	 *            it looks for faces in this frame
	 */
	private void detectAndDisplay(Mat frame)
	{
		MatOfRect faces = new MatOfRect();
		Mat grayFrame = new Mat();
		
		// convert the frame in gray scale
		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);
		
		// compute minimum face size (20% of the frame height, in our case)
		if (this.absoluteFaceSize == 0)
		{
			int height = grayFrame.rows();
			if (Math.round(height * 0.2f) > 0)
			{
				this.absoluteFaceSize = Math.round(height * 0.2f);
			}
		}
		
		// detect faces
		this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(
				this.absoluteFaceSize, this.absoluteFaceSize), new Size());
		
		// each rectangle in faces is a face: draw them!
		Rect[] facesArray = faces.toArray();
		for (int i = 0; i < facesArray.length; i++)
			Core.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
		
	}
	
	/**
	 * The action triggered by selecting the Haar Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void haarSelected(Event event)
	{
		// check whether the lpb checkbox is selected and deselect it
		if (this.lbpClassifier.isSelected())
			this.lbpClassifier.setSelected(false);
		
		this.checkboxSelection("resources/haarcascades/haarcascade_frontalface_alt.xml");
	}
	
	/**
	 * The action triggered by selecting the LBP Classifier checkbox. It loads
	 * the trained set to be used for frontal face detection.
	 */
	@FXML
	protected void lbpSelected(Event event)
	{
		// check whether the haar checkbox is selected and deselect it
		if (this.haarClassifier.isSelected())
			this.haarClassifier.setSelected(false);
		
		this.checkboxSelection("resources/lbpcascades/lbpcascade_frontalface.xml");
	}
	
	/**
	 * Method for loading a classifier trained set from disk
	 * 
	 * @param classifierPath
	 *            the path on disk where a classifier trained set is located
	 */
	private void checkboxSelection(String classifierPath)
	{
		// load the classifier(s)
		this.faceCascade.load(classifierPath);
		
		// now the video capture can start
		this.cameraButton.setDisable(false);
	}
	
	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 * 
	 * @param frame
	 *            the {@link Mat} representing the current frame
	 * @return the {@link Image} to show
	 */
	private Image mat2Image(Mat frame)
	{
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Highgui.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}
	
}
