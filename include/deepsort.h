#include <Eigen/Cholesky>
#include <map>

class KalmanFilter {
   public:
    // sisyphus
    static const double KalmanFilter::chi2inv95[10] = {0, 3.8415, 5.9915, 7.8147, 9.4877, 11.070, 12.592, 14.067, 15.507, 16.919};

    KalmanFilter() {
        int ndim = 4;
        double dt = 1.;

        _motion_mat = Eigen::MatrixXf::Identity(8, 8);
        for (int i = 0; i < ndim; i++) {
            _motion_mat(i, ndim + i) = dt;
        }
        _update_mat = Eigen::MatrixXf::Identity(4, 8);

        this->_std_weight_position = 1. / 20;
        this->_std_weight_velocity = 1. / 160;
    }

    KAL_DATA initiate(const DETECTBOX& measurement) {
        DETECTBOX mean_pos = measurement;
        DETECTBOX mean_vel;
        for (int i = 0; i < 4; i++) mean_vel(i) = 0;

        KAL_MEAN mean;
        for (int i = 0; i < 8; i++) {
            if (i < 4)
                mean(i) = mean_pos(i);
            else
                mean(i) = mean_vel(i - 4);
        }

        KAL_MEAN std;
        std(0) = 2 * _std_weight_position * measurement[3];
        std(1) = 2 * _std_weight_position * measurement[3];
        std(2) = 1e-2;
        std(3) = 2 * _std_weight_position * measurement[3];
        std(4) = 10 * _std_weight_velocity * measurement[3];
        std(5) = 10 * _std_weight_velocity * measurement[3];
        std(6) = 1e-5;
        std(7) = 10 * _std_weight_velocity * measurement[3];

        KAL_MEAN tmp = std.array().square();
        KAL_COVA var = tmp.asDiagonal();
        return std::make_pair(mean, var);
    }

    void predict(KAL_MEAN& mean, KAL_COVA& covariance) {
        // revise the data;
        DETECTBOX std_pos;
        std_pos << _std_weight_position * mean(3), _std_weight_position * mean(3), 1e-2, _std_weight_position * mean(3);
        DETECTBOX std_vel;
        std_vel << _std_weight_velocity * mean(3), _std_weight_velocity * mean(3), 1e-5, _std_weight_velocity * mean(3);
        KAL_MEAN tmp;
        tmp.block<1, 4>(0, 0) = std_pos;
        tmp.block<1, 4>(0, 4) = std_vel;
        tmp = tmp.array().square();
        KAL_COVA motion_cov = tmp.asDiagonal();
        KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
        KAL_COVA covariance1 = this->_motion_mat * covariance * (_motion_mat.transpose());
        covariance1 += motion_cov;

        mean = mean1;
        covariance = covariance1;
    }

    KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance) {
        DETECTBOX std;
        std << _std_weight_position * mean(3), _std_weight_position * mean(3), 1e-1, _std_weight_position * mean(3);
        KAL_HMEAN mean1 = _update_mat * mean.transpose();
        KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
        Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
        diag = diag.array().square().matrix();
        covariance1 += diag;
        //    covariance1.diagonal() << diag;
        return std::make_pair(mean1, covariance1);
    }

    KAL_DATA update(const KAL_MEAN& mean, const KAL_COVA& covariance, const DETECTBOX& measurement) {
        KAL_HDATA pa = project(mean, covariance);
        KAL_HMEAN projected_mean = pa.first;
        KAL_HCOVA projected_cov = pa.second;

        // chol_factor, lower =
        // scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        // kalmain_gain =
        // scipy.linalg.cho_solve((cho_factor, lower),
        // np.dot(covariance, self._upadte_mat.T).T,
        // check_finite=False).T
        Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
        Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose();  // eg.8x4
        Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean;                 // eg.1x4
        auto tmp = innovation * (kalman_gain.transpose());
        KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
        KAL_COVA new_covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
        return std::make_pair(new_mean, new_covariance);
    }

    Eigen::Matrix<float, 1, -1> gating_distance(const KAL_MEAN& mean, const KAL_COVA& covariance,
                                                const std::vector<DETECTBOX>& measurements, bool only_position) {
        KAL_HDATA pa = this->project(mean, covariance);
        if (only_position) {
            printf("not implement!");
            exit(0);
        }
        KAL_HMEAN mean1 = pa.first;
        KAL_HCOVA covariance1 = pa.second;

        //    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
        DETECTBOXSS d(measurements.size(), 4);
        int pos = 0;
        for (DETECTBOX box : measurements) {
            d.row(pos++) = box - mean1;
        }
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
        Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
        auto zz = ((z.array()) * (z.array())).matrix();
        auto square_maha = zz.colwise().sum();
        return square_maha;
    }

   private:
    Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
    Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
    float _std_weight_position;
    float _std_weight_velocity;
};

#define INFTY_COST 1e5
// for matching;
class linear_assignment {
    linear_assignment();
    linear_assignment(const linear_assignment&);
    linear_assignment& operator=(const linear_assignment&);
    static linear_assignment* instance;

   public:
    static linear_assignment* getInstance();
    TRACHER_MATCHD matching_cascade(tracker* distance_metric, tracker::GATED_METRIC_FUNC distance_metric_func, float max_distance,
                                    int cascade_depth, std::vector<Track>& tracks, const DETECTIONS& detections,
                                    std::vector<int>& track_indices, std::vector<int> detection_indices = std::vector<int>());
    TRACHER_MATCHD min_cost_matching(tracker* distance_metric, tracker::GATED_METRIC_FUNC distance_metric_func, float max_distance,
                                     std::vector<Track>& tracks, const DETECTIONS& detections, std::vector<int>& track_indices,
                                     std::vector<int>& detection_indices);
    DYNAMICM gate_cost_matrix(KalmanFilter* kf, DYNAMICM& cost_matrix, std::vector<Track>& tracks, const DETECTIONS& detections,
                              const std::vector<int>& track_indices, const std::vector<int>& detection_indices,
                              float gated_cost = INFTY_COST, bool only_position = false);
};

class tracker {
   public:
    NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;

    KalmanFilter* kf;

    int _next_idx;

   public:
    std::vector<Track> tracks;
    tracker(/*NearNeighborDisMetric* metric,*/
            float max_cosine_distance, int nn_budget, float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);
    void predict();
    void update(const DETECTIONS& detections);
    typedef DYNAMICM (tracker::*GATED_METRIC_FUNC)(std::vector<Track>& tracks, const DETECTIONS& dets,
                                                   const std::vector<int>& track_indices, const std::vector<int>& detection_indices);

   private:
    void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
    void _initiate_track(const DETECTION_ROW& detection);

   public:
    DYNAMICM gated_matric(std::vector<Track>& tracks, const DETECTIONS& dets, const std::vector<int>& track_indices,
                          const std::vector<int>& detection_indices);
    DYNAMICM iou_cost(std::vector<Track>& tracks, const DETECTIONS& dets, const std::vector<int>& track_indices,
                      const std::vector<int>& detection_indices);
    Eigen::VectorXf iou(DETECTBOX& bbox, DETECTBOXSS& candidates);
};

#ifndef TRACK_H
#define TRACK_H

#include "dataType.h"
#include "kalmanfilter.h"
#include "model.h"

class Track {
    enum TrackState { Tentative = 1, Confirmed, Deleted };

   public:
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int n_init, int max_age, const FEATURE& feature);
    void predit(KalmanFilter* kf);
    void update(KalmanFilter* const kf, const DETECTION_ROW& detection);
    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();
    DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    FEATURESS features;
    KAL_MEAN mean;
    KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;

   private:
    void featuresAppendOne(const FEATURE& f);
};

class NearNeighborDisMetric {
   public:
    enum METRIC_TYPE { euclidean = 1, cosine };
    NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget);
    DYNAMICM distance(const FEATURESS& features, const std::vector<int>& targets);
    //    void partial_fit(FEATURESS& features, std::vector<int> targets, std::vector<int> active_targets);
    void partial_fit(std::vector<TRACKER_DATA>& tid_feats, std::vector<int>& active_targets);
    float mating_threshold;

   private:
    typedef Eigen::VectorXf (NearNeighborDisMetric::*PTRFUN)(const FEATURESS&, const FEATURESS&);
    Eigen::VectorXf _nncosine_distance(const FEATURESS& x, const FEATURESS& y);
    Eigen::VectorXf _nneuclidean_distance(const FEATURESS& x, const FEATURESS& y);

    Eigen::MatrixXf _pdist(const FEATURESS& x, const FEATURESS& y);
    Eigen::MatrixXf _cosine_distance(const FEATURESS& a, const FEATURESS& b, bool data_is_normalized = false);

   private:
    PTRFUN _metric;
    int budget;
    std::map<int, FEATURESS> samples;
};
