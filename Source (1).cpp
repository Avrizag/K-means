#include <ctime>     // for a random seed
#include <fstream>   // for file-reading
#include <iostream>  // for file-reading
#include <sstream>   // for file-reading
#include <vector>
#include <limits>
#include<chrono>
#include<valarray> // for valarray functions


#define min(x,y) ((x<y)?(x):(y))
#define max(x,y) ((x>y)?(x):(y))

using namespace std;



struct Point {
    valarray<double> vec;     // coordinates
    int cluster;     // no default cluster
    double minDist;



    Point(valarray<double>& y) :vec(move(y)), cluster(-1), minDist(DBL_MAX) {
    }

    double distance(Point p) {//vec dist from vec
        valarray<double>&& tmp = p.vec - vec;
        return (tmp * tmp).sum();
    }

    bool operator ==(Point p) {
        valarray<bool> bools = vec == p.vec;
        for (bool v : bools) {
            if (!v)return false;
        }
        return true;
    }
};

vector <int> kMeansClustering(vector<Point>& points, int k);
vector<Point> initcenter(int k, vector<Point>& points);
vector<Point> findnewcentroids(int k, vector<Point>& points);
void assintocluster(vector<Point>& centroids, vector<Point>& points);
double silluate(vector <Point>& points, int k);
vector <int> pointtocluster(vector<Point>& points);
vector<int> kmeans_shillhuate(vector <Point>& points, int max_k);

vector<Point> readdata() {
    vector<Point> points;
    string line;
    ifstream file("data_1_3.txt");

    while (getline(file, line)) { //line by line
        stringstream lineStream(line);//work on line
        string bit;
        valarray<double> vec(100);
        int i = 0;
        while (getline(lineStream, bit, ',')) {//read word until ,
            vec[i++] = stof(bit); //string to float..
        }


        getline(lineStream, bit, '\n'); //read last word in line
        vec[99] = stof(bit);

        points.push_back(Point(vec));
    }
    points.shrink_to_fit(); //free unused space
    return move(points);
}
int main() {
    vector<Point> points = readdata();
    cout << "data loaded" << endl;
    auto t1 = chrono::high_resolution_clock::now();
    vector<int> out = kMeansClustering(points, 10); // pass address of points to function
    auto t2 = chrono::high_resolution_clock::now();
    cout << "finished Kmeans - time:"<< chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << endl;
    cout << silluate(points, 10) << endl;
}
vector <int> kMeansClustering(vector<Point>& points, int k) {
    vector <Point> centroids = initcenter(k, points);
    while (true) {
        assintocluster(centroids, points);
        vector<Point> newMeans = findnewcentroids(k, points);
        if (equal(newMeans.begin(), newMeans.end(), centroids.begin())) {
            return pointtocluster(points);
        }
        centroids = newMeans;
    }


}
vector <int> pointtocluster(vector<Point>& points) //create from data array with the index of point to which cluster it belongs.
{
    vector <int> ret;
    for (Point p : points) {
        ret.push_back(p.cluster);
    }
    ret.shrink_to_fit();
    return move(ret);
}
int argmax(vector<double>& vec) //index of max arg
{
    double max = DBL_MIN;
    int indexmax = -1;
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] > max)
        {
            max = vec[i];
            indexmax = i;
        }
    }
    return indexmax;
}

int getRandIndexWithWeights(vector <double>& weights) { //return the random 1 5 2 6  more weight more chance
    double sum = 0;
    for (double v : weights)
    {
        sum += v;
    }
    double delta = 1;
    double random = 0;
    if (sum > INT_MAX) {//if the sum is more then int max, reduce all weights by the same factor
        delta = INT_MAX / sum;
        random = rand();
    }
    else
        random = rand() % ((int)sum);
    for (int i = 0; i < weights.size(); i++)
    {
        random -= weights[i] * delta; // each index have different probability depands on the value 
        if (random < 0)return i;
    }
}

vector<Point> initcenter(int k, vector<Point>& points)//init for k means ++
{
    srand(time(NULL));
    int n = points[0].vec.size();
    int m = points.size();
    vector<Point> centroids;
    centroids.push_back(points.at(rand() % m));
    for (int j = 1; j < k; j++)
    {
        vector <double> Dtocenter_dist;
        for (int i = 0; i < points.size(); i++)
        {
            double mindist = points[i].distance(centroids[0]);
            for (int t = 1; t < centroids.size(); t++) // go over points and find the min dist to centroids
            {
                double tmp = points[i].distance(centroids[t]);
                mindist = min(mindist, tmp);
            }
            Dtocenter_dist.push_back(mindist); //push each dist into Dtocenter

        }
        centroids.push_back(points[getRandIndexWithWeights(Dtocenter_dist)]);// choose the next centroid by weight probability
    }


    centroids.shrink_to_fit();
    return move(centroids);
}


void assintocluster(vector<Point>& centroids, vector<Point>& points) //going through the points and find the min distance to find thier cluster
{
    for (vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {

        for (vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {

            int clusterId = c - begin(centroids);// index of centroid
            double dist = it->distance(*c);
            if (dist < it->minDist) {
                it->minDist = dist;
                it->cluster = clusterId;
            }
        }
    }
}

vector<Point> findnewcentroids(int k, vector<Point>& points) {// calc the new centroids.
    vector<int> nPoints;
    vector<valarray<double>> sumvec;
    valarray<double> t(0.0, points[0].vec.size());
    // Initialise with zeroes
    for (int j = 0; j < k; ++j) {
        nPoints.push_back(0);
        sumvec.push_back(t);
    }
    // go over points to append thier data to their centroids
    for (vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {
        int clusterId = it->cluster;
        nPoints[clusterId] += 1;
        sumvec[clusterId] += it->vec;//insert data for cluster

        it->minDist = DBL_MAX;  // reset distance for next iteration
    }
    // Compute the new centroids
    vector<Point> c;
    valarray<double> s;
    for (int j = 0; j < k; j++) {
        s = sumvec[j] / nPoints[j];
        Point b = Point(s);
        b.cluster = j;
        c.push_back(b);

    }
    c.shrink_to_fit();
    return move(c);




}
double silluate(vector <Point>& points, int k) { //shilluate 
    int m = points.size();
    vector<Point> means = findnewcentroids(k, points);
    double silSum = 0;
    for (Point p : points) { //go over points (shilluate for each point)
        double b = DBL_MAX; //find the distance for the closest cluster (not your own)
        double a = p.distance(means[p.cluster]);//distance p from his cluster centroid (mean)
        for (int i = 0; i < k; i++) { //go over k ckusters
            if (i != p.cluster) { //not the same cluster
                double tmp = p.distance(means[i]);
                b = min(b, tmp); //if smaller .
            }
        }
        silSum += ((b - a) / max(b, a)); //shilluate
    }
    return silSum / m;
}

vector<int> kmeans_shillhuate(vector <Point>& points, int max_k) {
    vector<int> outs;
    vector<vector<int>> kmeansres;
    vector<double> scores;
    for (int i = 2; i < max_k; i++) {

        outs = kMeansClustering(points, i);
        kmeansres.push_back(outs);
        scores.push_back(silluate(points, i));
    }
    int index = argmax(scores);
    return kmeansres[index];


}