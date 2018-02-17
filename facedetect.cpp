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

  void mergerects(vector<Rect>& detects, vector<Rect2d>& tracks, vector<Rect2d>& merged)
  {
    cout << "Track size before:: " << tracks.size() << endl;
    for(Rect r : detects)
    {
      Rect2d r2d(r);

      for(vector<Rect2d>::iterator it = tracks.begin(); it != tracks.end(); ++it)
      {
        Rect2d inter = r2d & (*it);
        if(r2d.area() * 0.7 < inter.area())
        {
          cout << "Deleting rect" << endl;
          tracks.erase(it);
          --it;
        }
      }
      merged.push_back(r2d);
    }

    cout << "Track size after:: " << tracks.size() << endl;
    if(tracks.size() > 0)
      merged.insert(merged.end(), tracks.begin(), tracks.end());
  }

  void detect(Mat& rawImg, Mat& faceImg)
  {
    Mat smallImg;

    resize(rawImg, smallImg, Size(rawImg.cols/nScale, rawImg.rows/nScale));
    cvtColor(smallImg, smallImg, COLOR_BGR2GRAY );


    if(faces.size() == 0 || nFrame%nTrackPeriod == 0)
    {
      nFrame = 1;
      cout << "Lets find faces" << endl;
      vector<Rect> faceInt;

      facedetect(smallImg, faceInt);

      if(faceInt.size() > 0)
      {
        vector<Rect2d> tracks;
        tracks.insert(tracks.end(), faces.begin(), faces.end());
        faces.erase(faces.begin(), faces.end());

        mergerects(faceInt, tracks, faces);
      }

      for(Rect f : faces)
        tracker->init(smallImg, f);

    }
    else
    {
      nFrame ++;
      for(Rect2d& face : faces)
      {
        tracker->update(smallImg, face);
      }
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
  VideoCapture cap(0);
  if(!cap.isOpened())
    return -1;

  vector<Mat> frames;
  FaceDetector *face = new FaceDetector(30, 2);

  namedWindow("faces");

  while(cap.grab())
  {
    Mat rawImg;
    Mat faceImg;

    cap.retrieve(rawImg);
    resize(rawImg, rawImg, Size(640,480));

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
