#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"

using namespace std;
using namespace cv;

class FaceDetector{
  private:
  CascadeClassifier cascade;
  int nFrame, nTrackPeriod, nScale;

  Ptr<Tracker> tracker;
  vector<Rect2d> faces;

  public:
  FaceDetector(int _nTrack, float _nScale)
  {
    nFrame = 0;
    nScale = _nScale;
    cascade.load("haarcascade_frontalface_alt2.xml");

    nTrackPeriod = _nTrack;
    tracker = TrackerKCF::create();
  }

  void detect(Mat& rawImg, Mat& faceImg)
  {
    Mat smallImg;

    resize(rawImg, smallImg, Size(rawImg.cols/nScale, rawImg.rows/nScale));
    cvtColor(smallImg, smallImg, COLOR_BGR2GRAY );

    if(faces.size() == 0)
    {
      cout << "Lets find faces" << endl;
      vector<Rect> faceInt;

      facedetect(smallImg, faceInt);
      if(faceInt.size() > 0)
      {
        for(Rect face : faceInt)
        {
          const Rect2d f(face);
          tracker->init(smallImg, f);
          faces.push_back(Rect2d(face));
        }
      }
    }
    else
    {
      vector<Rect2d> newFaces;
      for(Rect2d& face : faces)
      {
        Rect2d f(face);
        tracker->update(smallImg, f);
        newFaces.push_back(Rect2d(f));
      }
      faces.erase(faces.begin(), faces.end());
      faces.insert(faces.begin(), newFaces.begin(), newFaces.end());
    }

    faceImg = rawImg.clone();
    draw(faceImg, faces, nScale);
  }

  void draw(Mat& img, vector<Rect2d>& faces, float scale)
  {
    for(Rect2d r2d : faces)
    {
      Rect r(r2d);
      r.x = r.x * scale;
      r.y = r.y * scale;
      r.width = r.width * scale;
      r.height = r.height * scale;

      rectangle(img, r, Scalar( 255, 0, 0 ), 2, 1 );
    }
  }

  void facedetect(Mat& rawImg, vector<Rect> &faces)
  {
    Mat gray, smallImg;
    equalizeHist(rawImg, smallImg);

    cascade.detectMultiScale(smallImg, faces, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    cout << faces.size() << " faces found"<< endl;
  }


};

int main(int, char**)
{
  VideoCapture cap("face.mp4");
  if(!cap.isOpened())
    return -1;

  vector<Mat> frames;
  FaceDetector *face = new FaceDetector(10, 2);

  namedWindow("faces");

  while(cap.grab())
  {
    Mat img;
    cap.retrieve(img);
    resize(img, img, Size(640,480));
    frames.push_back(img);
  }

  for(Mat rawImg : frames)
  {
    Mat faceImg;
    int64 t0 = cv::getTickCount();
    face->detect(rawImg, faceImg);
    cout << "Time taken: " << (cv::getTickCount() - t0)/cv::getTickFrequency() << endl;

    imshow("faces", faceImg);
    int k = waitKey(1);
    if(k == 27)
    {
      break;
    }
  }
}
