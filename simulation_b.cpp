#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "Eigen/Dense"
#include "geometry.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#define SOKOL_IMPL
#define SOKOL_GLCORE33
#include "sokol_gfx.h"
#include "sokol_app.h"
#include "sokol_glue.h"
#include "imgui.h"
#include "util/sokol_imgui.h"
#include "imgui_overlayable.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "tag_mapper.h"

using namespace carl;

const double tag_side_length = 0.030;

struct StbImage {
    int x;
    int y;
    int n;
    uint8_t *data = nullptr;
};

struct AppState {
    bool images_loaded = false;
    bool tag_image_points_loaded = false;
    std::unordered_map<int, StbImage> images;
    std::map<int, sg_image> sg_images;
    std::unordered_map<int, std::unordered_map<int, std::array<Eigen::VectorD<2>, 4>>> tag_image_points;
    TagMapper tag_mapper;
    std::unordered_set<int> images_to_add;
    bool tag_mapper_loaded = false;
    Eigen::VectorD<4> camparams;
    int num_tags_added = 0;

    double last_err = 0;

    bool optimize = false;
    bool show_detects = false;
    bool show_projects = false;
};

std::array<Eigen::VectorD<4>, 4> make_tag_corners(double side_length) {
    const double l_2 = side_length/2; //
    Eigen::VectorD<4> tl = { -l_2, l_2, 0, 1 };
    Eigen::VectorD<4> tr = { l_2, l_2, 0, 1 };
    Eigen::VectorD<4> br = { l_2, -l_2, 0, 1 };
    Eigen::VectorD<4> bl = { -l_2, -l_2, 0, 1 };
    return { tl, tr, br, bl };
}

template <typename T>
std::string eigen_to_string(T&& m) {
    std::string out;
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::stringstream ss;
    ss << m.format(CleanFmt);
    return ss.str();
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

void load_tag_image_points(AppState* app) {
    assert(!app->tag_image_points_loaded);
    std::string buf;
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
                app->tag_image_points[i][tag_id] = img_points;
                // std::cout << "tag " << tag_id << ", img " << i << "\n";
                // for (int j = 0; j < 4; ++j) {
                //     std::cout << "\tpt" << j << ": " << img_points[j].transpose() << "\n";
                // }
            }
        }
    }

    app->tag_image_points_loaded = true;
}

void load_images(AppState* app) {
    assert(!app->images_loaded);

    for (int i = 0; i <= 14; ++i) {
        std::string image_path = absl::StrFormat("/home/mark/carl/example_data/image_%d.png", i);
        StbImage image;
        image.data = stbi_load(image_path.c_str(), &image.x, &image.y, &image.n, 4);
        app->images.emplace(std::make_pair(i, std::move(image)));
    }

    for (auto&& [id, cv_img] : app->images) {
        sg_image_desc img_desc {};
        img_desc.width = cv_img.x;
        img_desc.height = cv_img.y;
        img_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
        img_desc.data.subimage[0][0].ptr = cv_img.data;
        img_desc.data.subimage[0][0].size = cv_img.x * cv_img.y * 4;
        app->sg_images.insert(std::make_pair(id, sg_make_image(&img_desc)));
    }

    app->images_loaded = true;
}

Eigen::VectorD<4> load_camparams() {
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
    return camparams;
}

void load_tag_mapper(AppState* app) {
    assert(app->tag_image_points_loaded);
    assert(!app->tag_mapper_loaded);

    const auto& tag_image_points = app->tag_image_points;
    app->camparams = load_camparams();
    auto& camparams = app->camparams;

    app->tag_mapper.config().default_tag_side_length = tag_side_length;
    app->tag_mapper.config().default_camparams = camparams;

    TagMapper& tag_mapper = app->tag_mapper;    
    tag_mapper.init_camera(0, camparams);

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

    const auto& tags_to_add = tag_image_points.at(image_with_most_tags);
    for (auto& [tag_id, image_points] : tags_to_add) {
        std::cout << "tag " << tag_id << "\n";
        for (int j = 0; j < 4; ++j) {
            std::cout << "\tpt" << j << ": " << image_points[j].transpose() << "\n";
        }
        auto tx_camera_tag = initial_guess_tx_camera_tag(tag_side_length, camparams, image_points);
        tag_mapper.init_tag(tag_id, tx_camera_tag);
        tag_mapper.add_detection(/*camera_id=*/0, image_with_most_tags, tag_id, image_points);
        app->num_tags_added += 1;
    }

    for (auto& [image_id, _] : tag_image_points) {
        if (image_id == image_with_most_tags) {
            // already added this one
            continue;
        }
        app->images_to_add.insert(image_id);
    }

    app->tag_mapper_loaded = true;
}

void add_tag_mapper_image(AppState* app) {
    assert(app->tag_mapper_loaded);

    auto& images_to_add = app->images_to_add;
    auto& tag_image_points = app->tag_image_points;
    auto& tag_mapper = app->tag_mapper;

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
    auto tx_camera_tag = initial_guess_tx_camera_tag(tag_side_length, app->camparams, tags_to_add.at(best_tag));
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

        auto tx_camera_tag = initial_guess_tx_camera_tag(tag_side_length, app->camparams, img_points);
        auto tx_world_tag = tx_world_camera * tx_camera_tag;
        std::cout << "the tag from this camera view is at\n" << "\n";
        std::cout << tx_world_tag << "\n";

        // do projection and see what happens
        if (!have_tag) {
            auto tag_corners = make_tag_corners(tag_side_length);
            for (int i = 0; i < 4; ++i) {
                Eigen::VectorD<4> camera_point = tx_camera_tag * tag_corners[i];
                Eigen::VectorD<2> proj_point = apply_camera_matrix(app->camparams, camera_point);
                std::cout << "proj point[" << i <<"] " << proj_point.transpose() << "\n";
                std::cout << "img point[" << i <<"] " << img_points[i].transpose() << "\n";
            }

            tag_mapper.init_tag(tag_id, tx_world_tag);
        }
        tag_mapper.add_detection(/*camera_id=*/0, best_image_to_add, tag_id, img_points);
        app->num_tags_added += 1;
    }

    images_to_add.erase(best_image_to_add);
}

static struct {
    sg_pass_action pass_action;
} state;

void init_cb() {
    const sg_desc desc {.context = sapp_sgcontext()};
    sg_setup(&desc);
    const simgui_desc_t imgui_desc { 0 };
    simgui_setup(&imgui_desc);
    state.pass_action = {};
    state.pass_action.colors[0] = { .action = SG_ACTION_CLEAR, .value = { 0.0f, 0.5f, 1.0f, 1.0 } };
};

const uint32_t tag_colors[4] = {
    ImColor(1.0f, 0.0f, 0.0f, 1.0f),
    ImColor(0.0f, 1.0f, 0.0f, 1.0f),
    ImColor(0.0f, 0.0f, 1.0f, 1.0f),
    ImColor(1.0f, 1.0f, 0.0f, 1.0f)
};

void frame_cb() {
    static AppState app = {};
    if (!app.images_loaded) {
        std::cout << "Loading images" << "\n";
        load_images(&app);
    }
    if(!app.tag_image_points_loaded) {
        std::cout << "Loading tag image points" << "\n";
        load_tag_image_points(&app);
    }
    if(!app.tag_mapper_loaded) {
        std::cout << "Loading tag mapper" << "\n";
        load_tag_mapper(&app);
    }

    const simgui_frame_desc_t frame_desc {
        .width = sapp_width(),
        .height = sapp_height(),
        .delta_time = sapp_frame_duration(),
        .dpi_scale = sapp_dpi_scale(),
    };
    
    simgui_new_frame(&frame_desc);

    /*=== UI CODE STARTS HERE ===*/
    ImGui::SetNextWindowPos((ImVec2){10,10}, ImGuiCond_Once, (ImVec2){0,0});
    ImGui::SetNextWindowSize((ImVec2){400, 100}, ImGuiCond_Once);

    ImGui::Begin("Images", 0, ImGuiWindowFlags_None);
    const float window_width = ImGui::GetWindowWidth();
    ImGui::Text("%lu images", app.images.size());
    for (auto &&[id, img] : app.sg_images) {
        if (!app.tag_mapper.have_image(id)) {
            continue;
        }
        const auto info = sg_query_image_info(img);
        auto image = ImGuiOverlayable::Image((void*)img.id, info.width, info.height, window_width-10);

        if (app.show_detects) {
            for (auto&& [tag, pts] : app.tag_image_points[id]) {
                for (int idx =0; idx < int(pts.size()); ++idx) {
                    int next_idx = (idx + 1) % pts.size();
                    const auto& pt = pts[idx];
                    const auto& next_pt = pts[next_idx];
                    image.add_line(float(pt(0)), float(pt(1)),
                                   float(next_pt(0)), float(next_pt(1)),
                                   tag_colors[idx]);
                }
            }
        }

        const auto tx_world_camera = app.tag_mapper.get_camera_pose(id);
        if (app.show_projects) {
            const auto camparams = app.tag_mapper.get_camparams(0);        
            
            for (auto&& [tag, pts] : app.tag_image_points[id]) {
                if (!app.tag_mapper.have_tag(tag)) {
                    continue;
                }

                std::array<Eigen::VectorD<2>, 4> proj_points;

                // take the tag from the tagmapper
                const Eigen::SquareD<4> tx_world_tag = app.tag_mapper.get_tag_pose(tag);
                const Eigen::SquareD<4> tx_camera_tag = tx_world_camera.inverse() * tx_world_tag;
                const auto tag_corners = make_tag_corners(tag_side_length);
                for (int i = 0; i < 4; ++i) {
                    Eigen::VectorD<4> camera_point = tx_camera_tag * tag_corners[i];
                    Eigen::VectorD<2> proj_point = apply_camera_matrix(camparams, camera_point);
                    proj_points[i] = proj_point;
                }

                // project it
                for (int idx =0; idx < int(proj_points.size()); ++idx) {
                    int next_idx = (idx + 1) % proj_points.size();
                    const auto& pt = proj_points[idx];
                    const auto& next_pt = proj_points[next_idx];
                    image.add_line(float(pt(0)), float(pt(1)),
                                   float(next_pt(0)), float(next_pt(1)),
                                   tag_colors[idx]);
                }
            }
        }
    }
    ImGui::End();

    ImGui::Begin("Control");
    if (!app.images_to_add.empty() && ImGui::Button("Add image")) {
        add_tag_mapper_image(&app);
    }
    ImGui::Checkbox("Show Projects", &app.show_projects);
    ImGui::Checkbox("Show Detects", &app.show_detects);
    ImGui::Checkbox("Optimize", &app.optimize);
    const bool optimize_once = ImGui::Button("Optimize Once");
    if ((app.optimize || optimize_once) && app.tag_mapper_loaded) {
        app.last_err = app.tag_mapper.update();
    }
    ImGui::End();

    ImGui::Begin("Info");
    if (app.tag_mapper_loaded) {
        auto camparams = app.tag_mapper.get_camparams(0);
        ImGui::Text("camparams: %s", eigen_to_string(camparams.transpose()).c_str());
        ImGui::Text("last err: %f", app.last_err);
        ImGui::Text("last err stddev: %f", sqrt(app.last_err/app.num_tags_added));
    }
    ImGui::End();
    /*=== UI CODE ENDS HERE ===*/

    sg_begin_default_pass(&state.pass_action, sapp_width(), sapp_height());
    simgui_render();
    sg_end_pass();
    sg_commit();    
};

void cleanup_cb() {
    simgui_shutdown();
    sg_shutdown();
};

void event_cb(const sapp_event* event) {
    simgui_handle_event(event);
};

sapp_desc sokol_main(int argc, char* argv[]) {
    sapp_desc desc = {
        .init_cb = init_cb,
        .frame_cb = frame_cb,
        .cleanup_cb = cleanup_cb,
        .event_cb = event_cb,        
        .width = 1280,
        .height = 720,
    };
    return desc;
}
