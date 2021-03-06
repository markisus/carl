#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "Eigen/Dense"
#include "math/geometry.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#define SOKOL_IMPL
#define SOKOL_GLCORE33
#include "sokol/sokol_gfx.h"
#include "sokol/sokol_app.h"
#include "sokol/sokol_glue.h"
#include "imgui.h"
#include "sokol/util/sokol_imgui.h"
#include "util/imgui_overlayable.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "tag_mapper.h"
#include "tag_mapper_data.h"
#include "factor_graph/layout_data.h"
#include "util/image.h"
#include "util/thread_pool.h"
#include "entt/fwd.hpp"

using namespace carl;
using namespace tag_mapper;

ThreadPool thread_pool = {/*num_threads=*/1};

const int NON_TAG_ID = std::numeric_limits<int>::max();

struct AppState {
    bool images_loaded = false;
    bool tag_image_points_loaded = false;
    std::unordered_map<std::string, Image> images;
    TagMapper tag_mapper;
    std::unordered_set<std::string> images_to_add;
    bool tag_mapper_loaded = false;
    int num_tags_added = 0;

    double last_err = 0;
    double last_layout_err = 0;
    bool layout_converged = false;

    bool optimize = false;
    bool show_detects = false;
    bool show_projects = true;

    ImGuiOverlayable graph_display;

    std::string selected_image_id = "";
    int selected_tag_id = NON_TAG_ID;

    // camera control for 3d view
    float dx = 0;
    float dy = 0;
    float dz = 0;
    float droll = 0;
    float dpitch = 0;
    float dyaw = 0;
};

std::array<Eigen::VectorD<4>, 4> make_tag_corners(double side_length) {
    const double l_2 = side_length/2; //
    Eigen::VectorD<4> tl = { -l_2, l_2, 0, 1 };
    Eigen::VectorD<4> tr = { l_2, l_2, 0, 1 };
    Eigen::VectorD<4> br = { l_2, -l_2, 0, 1 };
    Eigen::VectorD<4> bl = { -l_2, -l_2, 0, 1 };
    return { tl, tr, br, bl };
}

Eigen::MatrixD<4> get_four_camera_rays(
    const double w, const double h,
    const Eigen::VectorD<4>& camparams) {
    const double fx = camparams(0);
    const double fy = camparams(1);
    const double cx = camparams(2);
    const double cy = camparams(3);
    Eigen::MatrixD<4> rays;
    rays.col(0) << -cx/fx, -cy/fy, 1, 1;
    rays.col(1) << (w-cx)/fx, -cy/fy, 1, 1;
    rays.col(2) << (w-cx)/fx, (h-cy)/fy, 1, 1;
    rays.col(3) << -cx/fx, (h-cy)/fy, 1, 1;
    return rays;
}

template <typename T>
std::string eigen_to_string(T&& m) {
    std::string out;
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::stringstream ss;
    ss << m.format(CleanFmt);
    return ss.str();
}

Eigen::MatrixD<4> initial_guess_tx_camera_tag(double tag_side_length,
                                              const Eigen::VectorD<4>& camparams,
                                              const std::array<Eigen::VectorD<2>, 4>& image_points) {
    const double fx = camparams(0);
    const double fy = camparams(1);
    const double cx = camparams(2);
    const double cy = camparams(3);

    Eigen::MatrixD<3> camera_matrix;
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

    Eigen::MatrixD<3> eigen_rmat;
    cv::cv2eigen(rmat, eigen_rmat);

    Eigen::VectorD<3> eigen_tvec;
    cv::cv2eigen(tvec, eigen_tvec);

    Eigen::MatrixD<4> tx_camera_tag;
    tx_camera_tag.block<3,3>(0,0) = eigen_rmat;
    tx_camera_tag.block<3,1>(0,3) = eigen_tvec;
    tx_camera_tag.row(3) << 0, 0, 0, 1;
    return tx_camera_tag;
}

void load_image_data(AppState* app_ptr) {
    AppState& app = *app_ptr;
    std::string image_just_added = app.tag_mapper.image_list().back();
    auto& image_data = app.images[image_just_added];

    // put a placholder for now
    int default_width = 10;
    int default_height = 5;
    for (const auto& [other_image_id, other_image] : app.images) {
        if (other_image.data) {
            default_width = other_image.width;
            default_height = other_image.height;
            break;
        }
    }
    image_data.width = default_width;
    image_data.height = default_height;

    // start a job to load the image
    auto& scene = app.tag_mapper.get_scene();
    if (scene.image_paths.count(image_just_added)) {
        const std::string& path = scene.image_paths.at(image_just_added);
        thread_pool.Push([path, &image_data](){
            image_data.load(path, /*make_texture=*/false);                                    
        });
    }
}

void load_tag_mapper(AppState* app) {
    assert(app->tag_image_points_loaded);
    assert(!app->tag_mapper.get_scene().tag_detections.empty());
    assert(!app->tag_mapper_loaded);

    TagMapper& tag_mapper = app->tag_mapper;
    Scene& scene = tag_mapper.get_scene();

    std::string image_with_most_tags;
    size_t most_tags = 0;
    for (auto& [image_id, tags] : scene.tag_detections) {
        // std::cout << "image " << image_id << " has " << tags.size() << " tags" << "\n";
        if (tags.size() >= most_tags) {
            image_with_most_tags = image_id;
            most_tags = tags.size();
        }
    }
    std::cout << "initializing with image with most tags: " << image_with_most_tags << ", with " << most_tags << " tags" << "\n";
    tag_mapper.init_image(image_with_most_tags, Eigen::id<4>().eval());

    const auto& tags_to_add = scene.tag_detections.at(image_with_most_tags);
    for (const auto& [tag_id, image_points] : tags_to_add) {
        // std::cout << "tag " << tag_id << "\n";
        // for (int j = 0; j < 4; ++j) {
        //     std::cout << "\tpt" << j << ": " << image_points[j].transpose() << "\n";
        // }
        auto tx_camera_tag = initial_guess_tx_camera_tag(scene.get_tag_side_length(tag_id), scene.camparams, image_points);
        tag_mapper.init_tag(tag_id, tx_camera_tag);
        tag_mapper.add_detection(image_with_most_tags, tag_id);
        app->num_tags_added += 1;
    }
    for (auto& [image_id, _] : scene.tag_detections) {
        if (image_id == image_with_most_tags) {
            // already added this one
            continue;
        }
        app->images_to_add.insert(image_id);
    }
    // tag_mapper.relinearize();
    app->tag_mapper_loaded = true;

    load_image_data(app);
}

void add_tag_mapper_image(AppState* app) {
    assert(app->tag_mapper_loaded);

    auto& images_to_add = app->images_to_add;
    auto& tag_mapper = app->tag_mapper;
    auto& scene = app->tag_mapper.get_scene();

    // std::cout << "Selecting an image to add" << "\n";
    std::string best_image_to_add;
    double min_covar_trace = std::numeric_limits<double>::infinity();
    int best_tag = -1;
    for (auto image_id : images_to_add) {
        // get mean, get covar for all tags that we
        for (auto& [tag_id, _] : scene.tag_detections[image_id]) {
            if (tag_mapper.have_tag(tag_id)) {
                Eigen::MatrixD<6> covariance;
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

    // std::cout << "best tag " << best_tag << " of image " << best_image_to_add << " with trace " << min_covar_trace << "\n";

    const auto& tags_to_add = scene.tag_detections[best_image_to_add];
    auto tx_world_tag = tag_mapper.get_tag_pose(best_tag);
    auto tx_camera_tag = initial_guess_tx_camera_tag(scene.get_tag_side_length(best_tag), scene.camparams, tags_to_add.at(best_tag));
    auto tx_world_camera = tx_world_tag * tx_camera_tag.inverse();
    tag_mapper.init_image(best_image_to_add, tx_world_camera);        
        
    for (auto& [tag_id, img_points] : tags_to_add) {
        // std::cout << "tag " << tag_id << "\n";
        // for (int j = 0; j < 4; ++j) {
        //     std::cout << "\tpt" << j << ": " << img_points[j].transpose() << "\n";
        // }

        bool have_tag = tag_mapper.have_tag(tag_id);
        if (have_tag) {
            // std::cout << "graph has this tag at\n" << "\n";
            // std::cout << tag_mapper.get_tag_pose(tag_id) << "\n";
        }

        auto tx_camera_tag = initial_guess_tx_camera_tag(scene.get_tag_side_length(tag_id), scene.camparams, img_points);
        auto tx_world_tag = tx_world_camera * tx_camera_tag;
        // std::cout << "the tag from this camera view is at\n" << "\n";
        // std::cout << tx_world_tag << "\n";

        // do projection and see what happens
        if (!have_tag) {
            // auto tag_corners = make_tag_corners(scene.get_tag_side_length(tag_id));
            // for (int i = 0; i < 4; ++i) {
            //     // Eigen::VectorD<4> camera_point = tx_camera_tag * tag_corners[i];
            //     // Eigen::VectorD<2> proj_point = apply_camera_matrix(scene.camparams, camera_point);
            //     // std::cout << "proj point[" << i <<"] " << proj_point.transpose() << "\n";
            //     // std::cout << "img point[" << i <<"] " << img_points[i].transpose() << "\n";
            // }

            tag_mapper.init_tag(tag_id, tx_world_tag);
        }
        tag_mapper.add_detection(best_image_to_add, tag_id);
        app->num_tags_added += 1;
    }

    images_to_add.erase(best_image_to_add);

    load_image_data(app);
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

const float MAX_DISPLAY_AGE = 100;

void display_variable(const VariableViz& viz, void* d) {
    AppState& app = *((AppState*)d);

    const float lens[2] = { 0.2, 0.15 };
    const float hues[2] = { 0.66, 0.0 };
    float lits[2] = { 1.0, 1.0 };
    float sats[2] = { 0, 0 };

    for (int i = 0; i < 2; ++i) {
        if (i == 0) {
            float age = (float)viz.age;
            age = std::min<float>(age/MAX_DISPLAY_AGE, 1);
            lits[i] = 1.0 - age;
        }
        if (i == 1) {
            float total_error = float(viz.error);
            const float error = std::sqrt(total_error);
            const float error_max = 20;
            const float redness = std::clamp<float>(error/error_max, 0, 1);
            sats[i] = redness;
        }

        auto picker = app.graph_display.add_circle_filled(
            (*viz.position)(0),
            (*viz.position)(1),
            lens[i]/2,
            ImColor::HSV(hues[i], sats[i], lits[i], 1.0f));

        if (i == 0) {
            if (picker.contains(ImGui::GetMousePos())) {
                ImGui::SetTooltip(
                    "%s %s\nError: %f\nAge: %d",
                    VAR_TYPE_STRS[viz.type],
                    viz.id->c_str(),
                    viz.error,
                    viz.age);

                if (ImGui::IsMouseClicked(0)) {
                    if (viz.type == VAR_TYPE_IMAGE_POSE) {
                        app.selected_image_id = *viz.id;
                    }
                    if (viz.type == VAR_TYPE_TAG_POSE) {
                        app.selected_tag_id = std::stoi(*viz.id); // very ugly... use std::any?
                    }
                }
            }
        }
    }
};

void display_factor(const FactorViz& viz, void* d) {
    AppState& app = *((AppState*)d);

    const float lens[2] = { 0.2, 0.15 };
    const float hues[2] = { 0.66, 0.0 };
    float lits[2] = { 1.0, 1.0 };
    float sats[2] = { 0, 0 };

    for (int i = 0; i < 2; ++i) {
        if (i == 0) {
            float age = (float)viz.age;
            age = std::min<float>(age/MAX_DISPLAY_AGE, 1);
            lits[i] = 1.0 - age;
        }
        if (i == 1) {
            float total_error = float(viz.error);
            if (total_error < 0) {
                std::cout << "Negative error? " << total_error << "\n";
                exit(-1);
            }
            const float error = std::sqrt(total_error);
            const float error_max = 20;
            const float redness = std::clamp<float>(error/error_max, 0, 1);
            sats[i] = redness;
        }

        auto picker = app.graph_display.add_rect_filled(
            (*viz.position)(0) - lens[i]/2,
            (*viz.position)(1) - lens[i]/2,
            (*viz.position)(0) + lens[i]/2,
            (*viz.position)(1) + lens[i]/2,
            ImColor::HSV(hues[i], sats[i], lits[i], 1.0f));

        if (i == 0) {            
            if (picker.contains(ImGui::GetMousePos())) {
                ImGui::SetTooltip("Tag %d, Image %s\nError: %f\nAge: %d",
                                  viz.tag_id,
                                  viz.image_id->c_str(),
                                  viz.error,
                                  viz.age);
                if (ImGui::IsMouseClicked(0)) {
                    app.selected_image_id = *viz.image_id;
                    app.selected_tag_id = viz.tag_id;
                }
            }
        }
    }
};

void display_edge(const EdgeViz& viz, void* d) {
    AppState& app = *((AppState*)d);

    Eigen::VectorD<2> center = (*viz.variable_position + *viz.factor_position)/2;
    const float to_factor_redness = std::min<float>(viz.to_factor_residual/0.1, 1.0f);
    const float to_var_redness = std::min<float>(viz.to_variable_residual/0.1, 1.0f);

    app.graph_display.add_line(
        (*viz.variable_position)(0),
        (*viz.variable_position)(1),
        center(0), center(1),
        ImColor::HSV(1.0f, to_var_redness, 1.0f, 0.3f + 0.7*to_var_redness),
        1.0);

    app.graph_display.add_line(
        (*viz.factor_position)(0),
        (*viz.factor_position)(1),
        center(0), center(1),
        ImColor::HSV(1.0f, to_factor_redness, 1.0f, 0.3f + 0.7*to_factor_redness),
        1.0);

    auto picker = app.graph_display.add_circle(
        center(0), center(1), 0.05,
        ImColor::HSV(1.0f, 0.0f, 1.0f, 0.3f));

    if (picker.contains(ImGui::GetMousePos())) {
        ImGui::SetTooltip("residuals\nto_factor: %f\nto_variable: %f",
                          viz.to_factor_residual,
                          viz.to_variable_residual);
    }
};

void frame_cb() {
    static AppState app = {};
    if (!app.images_loaded) {
        app.tag_mapper.load_scene("/home/mark/carl/example_data/");
        app.images_loaded = true;
        app.tag_image_points_loaded = true;
    }
    if(!app.tag_mapper_loaded) {
        std::cout << "Loading tag mapper" << "\n";
        load_tag_mapper(&app);
        app.tag_mapper_loaded = true;
    }

    auto& scene = app.tag_mapper.get_scene();

    const simgui_frame_desc_t frame_desc {
        .width = sapp_width(),
        .height = sapp_height(),
        .delta_time = sapp_frame_duration(),
        .dpi_scale = sapp_dpi_scale(),
    };
    
    simgui_new_frame(&frame_desc);

    /*=== UI CODE STARTS HERE ===*/

    ImGui::SetNextWindowPos(ImVec2{0,0});
    ImGui::SetNextWindowSize(ImVec2{frame_desc.width,frame_desc.height});
    ImGui::Begin("UI", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse);
    {
        ImGui::BeginChild("left pane", ImVec2(400, 0), true);
        {
            ImGui::Checkbox("Optimize", &app.optimize);
            if (app.optimize) {
                app.tag_mapper.update();
            }
            else {
                ImGui::SameLine();
                const bool optimize_once = ImGui::Button("Optimize Once");
                if ((app.optimize || optimize_once) && app.tag_mapper_loaded) {
                    app.last_err = app.tag_mapper.update();
                }
                ImGui::SameLine();
                const bool relinearize_once = ImGui::Button("Relinearize Once");
                if (relinearize_once && app.tag_mapper_loaded) {
                    app.tag_mapper.relinearize();
                }
            }

            auto camparams = app.tag_mapper.get_camparams();
            ImGui::Text("Camparams (fx,fy,cx,cy): %s", eigen_to_string(camparams.transpose()).c_str());
            ImGui::Text("Images (%lu / %lu)", app.tag_mapper.image_list().size(), scene.tag_detections.size());
            const auto& images = app.tag_mapper.image_list();
            if (ImGui::BeginListBox("##Images Selector", ImVec2(-FLT_MIN, 10*ImGui::GetTextLineHeightWithSpacing()))) {
                const bool is_selected = app.selected_image_id.empty();
                if (ImGui::Selectable("Select All", is_selected)) {
                    app.selected_image_id = "";
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                for (const auto& image_id : images) {
                    const bool is_selected = (app.selected_image_id == image_id);
                    if (ImGui::Selectable(image_id.c_str(), is_selected)) {
                        app.selected_image_id = image_id;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }
            ImGui::EndListBox();

            ImGui::Text("Tags");
            const auto& tags = app.tag_mapper.tag_list();
            if (ImGui::BeginListBox("##Tags Selector", ImVec2(-FLT_MIN, 10*ImGui::GetTextLineHeightWithSpacing()))) {
                const bool is_selected = app.selected_tag_id == NON_TAG_ID;
                if (ImGui::Selectable("Select All", is_selected)) {
                    app.selected_tag_id = NON_TAG_ID;
                }
                for (const int tag_id : tags) {
                    const bool is_selected = (app.selected_tag_id == tag_id);
                    if (ImGui::Selectable(std::to_string(tag_id).c_str(), is_selected)) {
                        app.selected_tag_id = tag_id;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
            }
            ImGui::EndListBox();
            
            if (!app.images_to_add.empty() && ImGui::Button("Add image")) {
                add_tag_mapper_image(&app);

            }
        }
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("right pane", ImVec2(0, 0), true);
        {
            int selected_tab = 0;            
            const float window_width = ImGui::GetWindowWidth();
            if (ImGui::BeginTabBar("right pane bar")) {
                if (ImGui::BeginTabItem("Image View")) {
                    ImGui::Checkbox("Show Projects", &app.show_projects); ImGui::SameLine(); ImGui::Checkbox("Show Detects", &app.show_detects);
                    selected_tab = 0;
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Graph View")) {
                    selected_tab = 1;

                    app.last_layout_err = app.tag_mapper.update_layout(&app.layout_converged);
                    ImGui::Text("Layout Err: %f", app.last_layout_err);
                    ImGui::SameLine();
                    ImGui::Text("Layout Converged: %s", app.layout_converged ? "yes" : "no");

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("3D View")) {
                    selected_tab = 2;

                    ImGui::SliderFloat("dx", &app.dx, -5, 5 );
                    ImGui::SliderFloat("dy",  &app.dy, -5, 5);
                    ImGui::SliderFloat("dz", &app.dz, -0.5, 5);
                    ImGui::SliderFloat("droll",  &app.droll, -1.0, 1.0);
                    ImGui::SliderFloat("dpitch",  &app.dpitch, -1.0, 1.0);
                    ImGui::SliderFloat("dyaw",  &app.dyaw, -1.0, 1.0);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }

            ImGui::BeginChild("tab contents");
            if (selected_tab == 0) {
                // num images per line depends on if we're viewing all, or 1 image
                int num_images_per_line = app.selected_image_id.empty() ? 4 : 1;
                int image_idx = 1;

                if (!app.selected_image_id.empty() &&
                    app.selected_tag_id != NON_TAG_ID) {
                    if(!scene.tag_detections[app.selected_image_id].count(app.selected_tag_id)) {
                        ImGui::Text("tag:%d does not appear in image:%s",
                                    app.selected_tag_id, app.selected_image_id.c_str());

                        if (ImGui::Button("View all tags in this image")) {
                            app.selected_tag_id = NON_TAG_ID;
                        }

                        if (ImGui::Button("View all images containing this tag")) {
                            app.selected_image_id = "";
                        }
                    }
                }
                
                for (const auto& id : app.tag_mapper.image_list()) {
                    if (!app.selected_image_id.empty() &&
                        app.selected_image_id != id) {
                        continue;
                    }

                    if (app.selected_tag_id != NON_TAG_ID) {
                        // don't display this image if it has none
                        // of the selected tags
                        if (!scene.tag_detections[id].count(app.selected_tag_id)) {
                            continue;
                        }
                    }
                    
                    if (!app.images.count(id)) {
                        // don't have this image yet... should not even be possible
                        continue;
                    }

                    auto& img_data = app.images[id];
                    if (!img_data.texture && img_data.data) {
                        // texture making must be done on main thread
                        img_data.make_texture();
                    }

                    ImGuiOverlayable image;
                    if (img_data.texture) {
                        image = ImGuiOverlayable::Image((void*)img_data.texture, img_data.width, img_data.height, (window_width-10)/num_images_per_line);
                    } else {
                        image = ImGuiOverlayable::Rectangle(img_data.width, img_data.height, (window_width-10)/num_images_per_line);
                    }
                    if ((image_idx % num_images_per_line) != 0) {
                        ImGui::SameLine();
                    }
                    image_idx += 1;

                    if (app.show_detects) {
                        for (auto&& [tag, pts] : scene.tag_detections[id]) {
                            image.add_text(float(pts[0](0)), float(pts[0](1)),
                                           ImColor::HSV(0.3f, 1.0, 1.0f, 1.0f),
                                           std::to_string(tag).c_str());
                
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

                    const auto tx_world_camera = app.tag_mapper.get_image_pose(id);
                    if (app.show_projects) {
                        const auto camparams = app.tag_mapper.get_camparams();        
            
                        for (auto&& [tag, pts] : scene.tag_detections[id]) {
                            if (!app.tag_mapper.have_tag(tag)) {
                                continue;
                            }

                            bool is_tag_focused = true;
                            if (app.selected_tag_id != NON_TAG_ID) {
                                is_tag_focused = (app.selected_tag_id == tag);
                            }
                            
                            std::array<Eigen::VectorD<2>, 4> proj_points;

                            // take the tag from the tagmapper
                            const Eigen::MatrixD<4> tx_world_tag = app.tag_mapper.get_tag_pose(tag);
                            const Eigen::MatrixD<4> tx_camera_tag = tx_world_camera.inverse() * tx_world_tag;
                            const auto tag_corners = make_tag_corners(scene.get_tag_side_length(tag));
                            for (int i = 0; i < 4; ++i) {
                                Eigen::VectorD<4> camera_point = tx_camera_tag * tag_corners[i];
                                Eigen::VectorD<2> proj_point = apply_camera_matrix(camparams, camera_point);
                                proj_points[i] = proj_point;
                            }

                            // project it
                            float thickness = 1.0;
                            if (is_tag_focused) {
                                thickness = 2.0;
                            }
                            for (int idx =0; idx < int(proj_points.size()); ++idx) {
                                int next_idx = (idx + 1) % proj_points.size();
                                ImColor color = tag_colors[idx];
                                if (!is_tag_focused) {
                                    color.Value.w = 0.5;
                                }
                                const auto& pt = proj_points[idx];
                                const auto& next_pt = proj_points[next_idx];
                                image.add_line(float(pt(0)), float(pt(1)),
                                               float(next_pt(0)), float(next_pt(1)),
                                               color, thickness);
                            }
                        }
                    }
                }
            }
            if (selected_tab == 1) {
                const float window_height = ImGui::GetWindowHeight();
                auto [width, height] = app.tag_mapper.layout_size();
                const double data_size = std::max(width, height) * 1.2;
                const double display_size = std::min(window_height-10, window_width-50);
                app.graph_display = ImGuiOverlayable::Rectangle(data_size, data_size, display_size);
                app.graph_display.data_center_x = 0;
                app.graph_display.data_center_y = 0;
                app.tag_mapper.visit_edge_viz(display_edge, (void*)(&app));        
                app.tag_mapper.visit_factor_viz(display_factor, (void*)(&app));
                app.tag_mapper.visit_variable_viz(display_variable, (void*)(&app));
            }
            if (selected_tab == 2) {
                const float window_height = ImGui::GetWindowHeight();
                const float f = 1000;
                
                Eigen::VectorD<4> camparams;
                camparams << f, f, window_width/2, window_height/2;

                if (!app.tag_mapper.image_list().empty()) {
                    Eigen::VectorD<6> dobject_se3;
                    dobject_se3 << -app.dpitch, app.dyaw, app.droll, 0, 0, 0;
                    dobject_se3.head<3>() *= M_PI;

                    Eigen::VectorD<6> dcamera_se3;
                    dcamera_se3 << 0, 0, 0, -app.dx, app.dy, app.dz;
                    const Eigen::MatrixD<4> dobject = se3_exp(dobject_se3);
                    const Eigen::MatrixD<4> dcamera = se3_exp(dcamera_se3);

                    Eigen::MatrixD<4> tx_tc_camera = Eigen::id<4>();
                    tx_tc_camera(2,3) = 0.5;
                    tx_tc_camera.col(2) *= -1;
                    tx_tc_camera.col(1) *= -1;
                           
                    Eigen::MatrixD<4> tx_world_tc = Eigen::id<4>();
                    if (!app.tag_mapper.tag_list().empty()) {
                        tx_world_tc = app.tag_mapper.get_tag_pose(app.tag_mapper.tag_list()[0]);
                    }
                    
                    auto image = ImGuiOverlayable::Rectangle(window_width, window_height, window_width-10);

                    // draw camera frustums
                    for (auto& frustum_id : app.tag_mapper.image_list()) {
                        // all cameras share the same camparams
                        if (!app.images.count(frustum_id)) {
                            std::cout << "skipping image " << frustum_id << " since don't have the image" << "\n";

                            // image not loaded yet,
                            // don't know width and height
                            continue;
                        }

                        ImColor frustum_color = ImColor::HSV(0.0f, 0.0f, 1.0f, 1.0f);

                        const bool is_image_focused = (frustum_id == app.selected_image_id || app.selected_image_id.empty());
                        if (!is_image_focused) {
                            frustum_color.Value.w = 0.5;
                        }

                        const auto& frustum_image = app.images[frustum_id];
                        const auto& frustum_camparams = scene.camparams;
                        const Eigen::MatrixD<4> rays = get_four_camera_rays(
                            frustum_image.width,
                            frustum_image.height,
                            frustum_camparams);

                        const Eigen::MatrixD<4> tx_tc_rays = tx_world_tc.inverse() * app.tag_mapper.get_image_pose(frustum_id);
                        const Eigen::MatrixD<4> tx_camera_rays = dcamera * tx_tc_camera.inverse() * dobject * tx_tc_rays;

                        // std::cout << "rays_camera \n" << rays_camera << "\n";

                        const double ray_scale_0 = 0.005;
                        const double ray_scale_1 = 0.01;

                        // top and bottom faces
                        for (double d : {ray_scale_0, ray_scale_1}) {
                            Eigen::MatrixD<4> rays_scaled = rays;
                            rays_scaled.block<3,4>(0,0) *= d;
                            const Eigen::MatrixD<4> rays_camera = tx_camera_rays * rays_scaled; // rays in 3d viewer frame

                            for (int i = 0; i < 4; ++i) {
                                int next_i = (i+1)%4;
                                Eigen::VectorD<4> ray_a = rays_camera.col(i);
                                Eigen::VectorD<4> ray_b = rays_camera.col(next_i);
                                Eigen::VectorD<2> ray_a_px = apply_camera_matrix(camparams, ray_a);
                                Eigen::VectorD<2> ray_b_px = apply_camera_matrix(camparams, ray_b);
                                image.add_line(float(ray_a_px(0)), float(ray_a_px(1)),
                                               float(ray_b_px(0)), float(ray_b_px(1)),
                                               frustum_color);
                            }
                        }

                        // links of top and bottom faces
                        for (int i = 0; i < 4; ++i) {
                            Eigen::MatrixD<4> rays_scaled_a = rays;
                            rays_scaled_a.block<3,4>(0,0) *= ray_scale_0;
                            Eigen::MatrixD<4> rays_scaled_b = rays;
                            rays_scaled_b.block<3,4>(0,0) *= ray_scale_1;
                            const Eigen::MatrixD<4> rays_camera_a = tx_camera_rays * rays_scaled_a; // rays in 3d viewer frame
                            const Eigen::MatrixD<4> rays_camera_b = tx_camera_rays * rays_scaled_b; // rays in 3d viewer frame
                            
                            Eigen::VectorD<4> ray_a = rays_camera_a.col(i);
                            Eigen::VectorD<4> ray_b = rays_camera_b.col(i);
                            Eigen::VectorD<2> ray_a_px = apply_camera_matrix(camparams, ray_a);
                            Eigen::VectorD<2> ray_b_px = apply_camera_matrix(camparams, ray_b);
                            image.add_line(float(ray_a_px(0)), float(ray_a_px(1)),
                                           float(ray_b_px(0)), float(ray_b_px(1)),
                                           frustum_color);
                        }
                    }

                    for (auto& tag : app.tag_mapper.tag_list()) {
                        bool is_tag_focused = true;
                        if (app.selected_tag_id != NON_TAG_ID) {
                            is_tag_focused = (app.selected_tag_id == tag);
                        }

                        // take the tag from the tagmapper
                        const Eigen::MatrixD<4> tx_tc_tag = tx_world_tc.inverse() * app.tag_mapper.get_tag_pose(tag);
                        const Eigen::MatrixD<4> tx_camera_tag = dcamera * tx_tc_camera.inverse() * dobject * tx_tc_tag;
                        const auto tag_corners = make_tag_corners(scene.get_tag_side_length(tag));
                        std::array<Eigen::VectorD<2>, 4> proj_points;                        
                        for (int i = 0; i < 4; ++i) {
                            Eigen::VectorD<4> camera_point = tx_camera_tag * tag_corners[i];
                            Eigen::VectorD<2> proj_point = apply_camera_matrix(camparams, camera_point);
                            proj_points[i] = proj_point;
                        }
                        // project it
                        float thickness = 1.0;
                        if (is_tag_focused) {
                            thickness = 2.0;
                        }
                        for (int idx =0; idx < int(proj_points.size()); ++idx) {
                            int next_idx = (idx + 1) % proj_points.size();
                            ImColor color = tag_colors[idx];
                            if (!is_tag_focused) {
                                color.Value.w = 0.5;
                            }
                            const auto& pt = proj_points[idx];
                            const auto& next_pt = proj_points[next_idx];
                            image.add_line(float(pt(0)), float(pt(1)),
                                           float(next_pt(0)), float(next_pt(1)),
                                           color, thickness);
                        }
                    }
                } else {
                    ImGui::Text("No data yet");
                }
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();
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
        .window_title = "Tag Mapper",
    };
    return desc;
}
