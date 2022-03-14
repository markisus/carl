#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "Eigen/Dense"
#include "geometry.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "tag_mapper.h"

using namespace carl;

std::array<Eigen::VectorD<4>, 4> make_tag_corners(double side_length) {
    const double l_2 = side_length/2; //
    Eigen::VectorD<4> tl = { -l_2, l_2, 0, 1 };
    Eigen::VectorD<4> tr = { l_2, l_2, 0, 1 };
    Eigen::VectorD<4> br = { l_2, -l_2, 0, 1 };
    Eigen::VectorD<4> bl = { -l_2, -l_2, 0, 1 };
    return { tl, tr, br, bl };
}

Eigen::SquareD<4> initial_guess_tx_camera_tag(double tag_side_length,
                                              const Eigen::VectorD<4>& camparams,
                                              const std::array<Eigen::VectorD<2>, 4>& image_points) {
    const double fx = camparams(0);
    const double fy = camparams(1);
    const double cx = camparams(2);
    const double cy = camparams(3);

    Eigen::SquareD<3> camera_matrix;
    camera_matrix <<
        fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    cv::Mat cv_camera_matrix;
    cv::eigen2cv(camera_matrix, cv_camera_matrix);
    
    std::vector<cv::Point2d> cv_image_points;
    std::vector<cv::Point3d> cv_obj_points;
    std::array<Eigen::VectorD<4>, 4> obj_points =  make_tag_corners(tag_side_length);
    for (const auto& image_point : image_points) {
        cv::Point2d cv_image_point;
        cv_image_point.x = image_point(0);
        cv_image_point.y = image_point(1);
        cv_image_points.push_back(cv_image_point);
    }
    for (const auto& obj_point : obj_points) {
        cv::Point3d cv_obj_point;
        cv_obj_point.x = obj_point(0);
        cv_obj_point.y = obj_point(1);
        cv_obj_point.z = obj_point(2);
        cv_obj_points.push_back(cv_obj_point);
    }
                
    std::vector<double> dist_coeffs;
    cv::Mat rvec;
    cv::Mat tvec;
    bool result = cv::solvePnP(cv_obj_points, cv_image_points, cv_camera_matrix, dist_coeffs, rvec, tvec);
    if (!result) {
        std::cout << "could not find initial guess\n";
        exit(-1);
    }

    cv::Mat rmat;
    cv::Rodrigues(rvec,rmat);

    Eigen::SquareD<3> eigen_rmat;
    cv::cv2eigen(rmat, eigen_rmat);

    Eigen::VectorD<3> eigen_tvec;
    cv::cv2eigen(tvec, eigen_tvec);

    Eigen::SquareD<4> tx_camera_tag;
    tx_camera_tag.block<3,3>(0,0) = eigen_rmat;
    tx_camera_tag.block<3,1>(0,3) = eigen_tvec;
    tx_camera_tag.row(3) << 0, 0, 0, 1;
    return tx_camera_tag;
}


int main(int argc, char *argv[])
{

    std::unordered_map<int, std::unordered_map<int, std::array<Eigen::VectorD<2>, 4>>> tag_image_points;

    // open file
    std::string buf;
    std::ifstream camera_mat_file("/home/mark/carl/example_data/camera_matrix.txt");
    Eigen::VectorD<4> camparams;
    for (int i = 0; i < 2; ++i) {
        std::getline(camera_mat_file, buf);
        const std::vector<absl::string_view> tokens = absl::StrSplit(buf, ' ');
        if (i == 0) {
            // store fx, cx
            bool result = 
                absl::SimpleAtod(tokens[0], &camparams(0)) &&
                absl::SimpleAtod(tokens[2], &camparams(2));
            if(!result) {
                std::cout << "parse error " << buf << "\n";
                exit(-1);
            }
        }
        if (i == 1) {
            // store fy, cy
            bool result = 
                absl::SimpleAtod(tokens[1], &camparams(1)) &&
                absl::SimpleAtod(tokens[2], &camparams(3));
            if(!result) {
                std::cout << "parse error " << buf << "\n";
                exit(-1);
            }
        }
    }
    std::cout << "camparams " << camparams.transpose() << "\n";

    const double tag_side_length = 0.030;
    
    for (int i = 0; i <= 14; ++i) {
        std::string tag_txt_path = absl::StrFormat("/home/mark/carl/example_data/tags_%d.txt", i);
        std::ifstream tag_file(tag_txt_path);
        std::cout << "tag txt " << tag_txt_path << "\n";
        int tag_id;
        std::array<Eigen::VectorD<2>, 4> img_points;
        for(int lcount = 0; std::getline(tag_file, buf); ++lcount) {
            if (lcount % 5 == 0) {
                if (absl::SimpleAtoi(buf, &tag_id)) {
                    // std::cout << "\tline " << lcount << ", tag " << tag_id << "\n";
                    continue;
                } else {
                    std::cout << "could not parse " << buf << "\n";
                    exit(-1);
                }
            }
            int point_idx = (lcount % 5) - 1;
            const std::vector<absl::string_view> tokens = absl::StrSplit(buf, ' ');
            double x, y;
            bool succ = absl::SimpleAtod(tokens[0], &x) && absl::SimpleAtod(tokens[1], &y);
            if (!succ) {
                std::cout << "could not parse " << buf << "\n";
                exit(-1);
            }
            img_points[point_idx] << x, y;
            if (point_idx == 3) {
                // last point of the square
                tag_image_points[i][tag_id] = img_points;
                // std::cout << "tag " << tag_id << ", img " << i << "\n";
                // for (int j = 0; j < 4; ++j) {
                //     std::cout << "\tpt" << j << ": " << img_points[j].transpose() << "\n";
                // }
            }
        }
    }

    TagMapperConfig config;
    config.default_tag_side_length = 0.030;
    config.default_camparams = camparams;
    
    TagMapper tag_mapper { std::move(config) };
    tag_mapper.init_camera(0, camparams);

    int detections_added = 0;
    std::set<int> tags_added;

    int image_with_most_tags = -1;
    size_t most_tags = 0;
    for (auto& [image_id, tags] : tag_image_points) {
        if (tags.size() >= most_tags) {
            image_with_most_tags = image_id;
            most_tags = tags.size();
        }
    }
    std::cout << "initializing with image with most tags: " << image_with_most_tags << ", with " << most_tags << " tags" << "\n";
    tag_mapper.init_image(image_with_most_tags, Eigen::id<4>().eval());
    const auto& tags_to_add = tag_image_points[image_with_most_tags];
    for (auto& [tag_id, image_points] : tags_to_add) {
        std::cout << "tag " << tag_id << "\n";
        for (int j = 0; j < 4; ++j) {
            std::cout << "\tpt" << j << ": " << image_points[j].transpose() << "\n";
        }
        auto tx_camera_tag = initial_guess_tx_camera_tag(tag_side_length, camparams, image_points);
        tag_mapper.init_tag(tag_id, tx_camera_tag);
        tags_added.insert(tag_id);

        tag_mapper.add_detection(/*camera_id=*/0, image_with_most_tags, tag_id, image_points);
        detections_added += 1;
    }
    for (int i = 0; i < 10; ++i) {
        double err2 = tag_mapper.update();
        std::cout << "it " << i << " error " << err2 << "\n";
    }
    
    std::unordered_set<int> images_to_add;
    for (auto& [image_id, _] : tag_image_points) {
        if (image_id == image_with_most_tags) {
            // already added this one
            continue;
        }
        images_to_add.insert(image_id);
    }

    while (!images_to_add.empty()) {
        std::cout << "Selecting an image to add" << "\n";
        int best_image_to_add = -1;
        double min_covar_trace = std::numeric_limits<double>::infinity();
        int best_tag = -1;
        for (auto image_id : images_to_add) {
            // get mean, get covar for all tags that we
            for (auto& [tag_id, _] : tag_image_points[image_id]) {
                if (tag_mapper.have_tag(tag_id)) {
                    Eigen::SquareD<6> covariance;
                    tag_mapper.get_tag_pose(tag_id, &covariance);
                    if (covariance.trace() < min_covar_trace) {
                        min_covar_trace = covariance.trace();
                        best_tag = tag_id;
                        best_image_to_add = image_id;
                    }
                }
            }
        }

        if (!std::isfinite(min_covar_trace)) {
            std::cout << "No overlap with existing map..." << "\n";
            exit(-1);
        }

        std::cout << "best tag " << best_tag << " of image " << best_image_to_add << " with trace " << min_covar_trace << "\n";
        const auto& tags_to_add = tag_image_points[best_image_to_add];
        auto tx_world_tag = tag_mapper.get_tag_pose(best_tag);
        auto tx_camera_tag = initial_guess_tx_camera_tag(tag_side_length, camparams, tags_to_add.at(best_tag));
        auto tx_world_camera = tx_world_tag * tx_camera_tag.inverse();
        tag_mapper.init_image(best_image_to_add, tx_world_camera);        
        
        for (auto& [tag_id, img_points] : tags_to_add) {
            std::cout << "tag " << tag_id << "\n";
            for (int j = 0; j < 4; ++j) {
                std::cout << "\tpt" << j << ": " << img_points[j].transpose() << "\n";
            }

            bool have_tag = tag_mapper.have_tag(tag_id);
            if (have_tag) {
                std::cout << "graph has this tag at\n" << "\n";
                std::cout << tag_mapper.get_tag_pose(tag_id) << "\n";
            }

            auto tx_camera_tag = initial_guess_tx_camera_tag(tag_side_length, camparams, img_points);
            auto tx_world_tag = tx_world_camera * tx_camera_tag;
            std::cout << "the tag from this camera view is at\n" << "\n";
            std::cout << tx_world_tag << "\n";

            // do projection and see what happens
            if (!have_tag) {
                tags_added.insert(tag_id);
                auto tag_corners = make_tag_corners(tag_side_length);
                for (int i = 0; i < 4; ++i) {
                    Eigen::VectorD<4> camera_point = tx_camera_tag * tag_corners[i];
                    Eigen::VectorD<2> proj_point = apply_camera_matrix(camparams, camera_point);
                    std::cout << "proj point[" << i <<"] " << proj_point.transpose() << "\n";
                    std::cout << "img point[" << i <<"] " << img_points[i].transpose() << "\n";
                }

                tag_mapper.init_tag(tag_id, tx_world_tag);
            }
            tag_mapper.add_detection(/*camera_id=*/0, best_image_to_add, tag_id, img_points);
            detections_added += 1;
            for (int i = 0; i < 3; ++i) {
                double err2 = tag_mapper.update();
                std::cout << "it " << i << " error2 " << err2 << " std " << std::sqrt(err2/detections_added) << "\n";
            }
        }

        for (int i = 0; i < 10; ++i) {
            double err2 = tag_mapper.update();
            std::cout << "it " << i << " error2 " << err2 << " std " << std::sqrt(err2/detections_added) << "\n";            
        }
        
        images_to_add.erase(best_image_to_add);
    }

    std::cout << "Final iterations\n";
    for (int i = 0; i < 500; ++i) {
        double err2 = tag_mapper.update();
        std::cout << "it " << i << " error2 " << err2 << " std " << std::sqrt(err2/detections_added) << "\n";                    
    }

    int home_tag = 9;
    Eigen::SquareD<4> tx_world_hometag = tag_mapper.get_tag_pose(home_tag);
    for (int tag_id : tags_added) {
        Eigen::SquareD<4> tx_world_tag = tag_mapper.get_tag_pose(tag_id);
        Eigen::SquareD<4> tx_hometag_tag = tx_world_hometag.inverse() * tx_world_tag;
        std::cout << "tx_world_tag (recentered) " << tag_id << "\n";
        std::cout << tx_hometag_tag  << "\n";
    }
    
    std::cout << "Hello world, goodbye!" << "\n";

    return 0;
}
