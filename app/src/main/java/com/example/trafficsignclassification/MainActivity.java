package com.example.trafficsignclassification;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    JavaCameraView javaCameraView;
    Mat mRGBA,mGRAY,mCANNY;
    private TextToSpeech mTTS;
    private String TAG="OPENCV";
    private final int thresh = 100;
    public NeuralNetwork nn;
    Long lastTime;
    TimerTask myTimerTask;
    Scalar color1 = new Scalar(255, 0, 0);
    Scalar color2 = new Scalar(255, 255, 0);
    Scalar color3 = new Scalar(0, 0, 255);
    List<String> signs= Arrays.asList("STOP", "Pedestrian crossing", "One-way road", "Parking", "Priority road", "Direct driving order", "Speed limit to 70 km / h");

    BaseLoaderCallback baseLoaderCallback= new BaseLoaderCallback(MainActivity.this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case BaseLoaderCallback.SUCCESS:{
                    javaCameraView.enableView();
                    break;}
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }

        }

    };

    void read(String name,Matrix matrix){
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open(name)));

            // do reading, usually loop until end of file reading
            String mLine;
            for (int i = 0; i < matrix.data.length; i++)
                for (int j = 0; j < matrix.data[i].length; j++){
                    mLine = reader.readLine();
                    matrix.data[i][j]= Double.parseDouble(mLine);
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d("OPENCV", "OpenCV ststus: "+ OpenCVLoader.initDebug());
        nn = new NeuralNetwork(900,200,7);
        read("weights_ho.dat",nn.weights_ho);
        read("weights_ih.dat",nn.weights_ih);
        read("bias_ih.dat",nn.bias_h);
        read("bias_ho.dat",nn.bias_o);
        lastTime = System.currentTimeMillis()/1000;
        javaCameraView=(JavaCameraView) findViewById(R.id.mycamera);
        //javaCameraView.setMaxFrameSize(100,200);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(MainActivity.this);
        mTTS = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int result = mTTS.setLanguage(Locale.ENGLISH);

                    if (result == TextToSpeech.LANG_MISSING_DATA
                            || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e("TTS", "Language not supported");
                    }
                } else {
                    Log.e("TTS", "Initialization failed");
                }
            }
        });
    }

    private void speak(String text) {

        float pitch = 0.7F;
        if (pitch < 0.1) pitch = 0.1f;
        float speed = 0.7F;
        if (speed < 0.1) speed = 0.1f;

        mTTS.setPitch(pitch);
        mTTS.setSpeechRate(speed);
        mTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null);
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        mRGBA= new Mat(width,height, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mRGBA.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        return processImage(inputFrame.rgba());
    }

    static double angle(Point pt1, Point pt2, Point pt0)
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    Point minPoint(MatOfPoint2f approx) {
        double x = 4000;
        double y = 4000;
        for (int i = 0; i < approx.toArray().length; i++) {
            if (approx.toArray()[i].x < x)
                x = approx.toArray()[i].x;
            if (approx.toArray()[i].y < y) {
                y = approx.toArray()[i].y;
            }
        }
        Point point= new Point(x,y);
        return point;
    }

    Point maxPoint(MatOfPoint2f approx) {
        double x = 0;
        double y = 0;
        for (int i = 0; i < approx.toArray().length; i++) {
            if (approx.toArray()[i].x > x)
                x = approx.toArray()[i].x;
            if (approx.toArray()[i].y > y) {
                y = approx.toArray()[i].y;
            }
        }
        Point point= new Point(x,y);
        return point;
    }

    private Mat cropMat(MatOfPoint2f dst){
        Point max,min;
        max=maxPoint(dst);
        min=minPoint(dst);
        Rect rect = new Rect(max,min);
        Mat ROI=new Mat(mGRAY,rect);
        Size size = new Size(30,30);
        Imgproc.resize( ROI, ROI, size );
        return  ROI;
    }

    private double [] MatToArray(Mat roi){
        //roi.convertTo(roi,CvType.CV_8UC1);
        byte [] image = new byte[(int) (roi.total() * roi.channels())];
        double [] out = new double[(int) (roi.total() * roi.channels())];
        roi.get(0,0,image);
        for(int i=0;i<image.length;i++)
            out[i]=(double) (image[i]+128)/256;
        return out;
    }

    private void detect( double [] image) {
        List<Double> scores=nn.predict(image);
        String text="";
        int point=-1;
        double tolerance=0.4;
        if(scores.get(0)>=tolerance)
           point=0;
        else if(scores.get(1)>=tolerance)
          point=1;
        else if(scores.get(2)>=tolerance)
            point=2;
        else if(scores.get(3)>=tolerance)
           point=3;
        else if(scores.get(4)>=tolerance)
           point=4;
        else if(scores.get(5)>=tolerance)
           point=5;
        else if(scores.get(6)>=tolerance)
           point=6;

        if(point!=-1)
        text=signs.get(point)+" confidence: "+(int)(scores.get(point)*100)+"%";
        Long actualTime = System.currentTimeMillis()/1000;
        if(text!="" && actualTime-lastTime>=4){
            speak(signs.get(point));
            lastTime=actualTime;
        }

        Imgproc.putText(mRGBA, text,
                new Point(mRGBA.cols() / 4, mRGBA.rows() / 2),
                Core.FONT_HERSHEY_COMPLEX, 0.5, color3);
    }

    public Mat processImage(Mat frame){
        mRGBA=frame;
        mGRAY=mRGBA.clone();
        mCANNY=mRGBA.clone();
        Imgproc.cvtColor(mRGBA,mGRAY,Imgproc.COLOR_RGBA2GRAY);
        Imgproc.blur(mGRAY,mGRAY,new Size(3,3));
        Imgproc.Canny(mGRAY,mCANNY,thresh,thresh*2);
        Imgproc.dilate(mCANNY,mCANNY,new Mat(),new Point(-1,-1),1);
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mCANNY,contours,hierarchy,Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint2f dst = new MatOfPoint2f();
            contours.get(i).convertTo(dst, CvType.CV_32FC2);
            Imgproc.approxPolyDP(dst, dst, 0.02 * Imgproc.arcLength(dst, true), true);
            double value = Imgproc.contourArea(dst);
            Vector<Double> cos= new Vector<>();
            if(value>1000)
                switch (dst.toArray().length) {
                    case 3: //trojkat metoda 1
                        cos.clear();
                        for (int j = 2; j <4; j++)
                            cos.add(angle(dst.toArray()[j % 3], dst.toArray()[j - 2], dst.toArray()[j - 1]));
                        Collections.sort(cos);
                        if ( cos.firstElement() >= 0.4 && cos.lastElement() <= 0.6){
                            Imgproc.drawContours(mRGBA, contours, i, color1, 2, Core.LINE_8, hierarchy, 0, new Point());
                            double [] image=MatToArray(cropMat(dst));
                            detect(image);

                        }

                        break;
                    case 4: //znaki kwadratowe
                        cos.clear();
                        for (int j = 2; j <5; j++)
                            cos.add(angle(dst.toArray()[j % 4], dst.toArray()[j - 2], dst.toArray()[j - 1]));
                        Collections.sort(cos);
                        if (  cos.firstElement() >= -0.1 && cos.lastElement() <= 0.3){
                            double [] image=MatToArray(cropMat(dst));
                            //nn.predict(image);
                           detect(image);

                            Imgproc.drawContours(mRGBA, contours, i, color2, 2, Core.LINE_8, hierarchy, 0, new Point());
                        }
                        else{// znaki trojkatne metoda 2
                            for (int j = 2; j <5; j++)
                                cos.add(angle(dst.toArray()[j % 4], dst.toArray()[j - 2], dst.toArray()[j - 1]));
                            Collections.sort(cos);
                            if ( cos.firstElement() >= -0.97 && cos.firstElement()<=-0.93 && cos.lastElement() <= 0.50 && cos.lastElement()>=0.40){
                                double [] image=MatToArray(cropMat(dst));
                                detect(image);
                            Imgproc.drawContours(mRGBA, contours, i, new Scalar(127, 0, 153), 2, Core.LINE_8, hierarchy, 0, new Point());

                            }
                        }

                        break;
                    case 8:// znaki okragle i stop
                        cos.clear();
                        for (int j = 2; j <9; j++)
                            cos.add(angle(dst.toArray()[j % 8], dst.toArray()[j - 2], dst.toArray()[j - 1]));
                        Collections.sort(cos);
                        if ( cos.firstElement() >= -0.8 && cos.lastElement() <= -0.6){
                            double [] image=MatToArray(cropMat(dst));

                        Imgproc.drawContours(mRGBA, contours, i, color3, 2, Core.LINE_8, hierarchy, 0, new Point());
                        detect(image);
                        }
                        break;
                    default:
                        //Imgproc.drawContours(mRGBA, contours, i, new Scalar(5, 190, 227), 2, Core.LINE_8, hierarchy, 0, new Point());
                        break;
                }
        }
        return  mRGBA;
    }

    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(javaCameraView != null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(javaCameraView != null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV is working.");
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
        else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,baseLoaderCallback);
        }
    }
}