struct CirclePicker {
    ImVec2 center;
    double radius;

    bool contains(ImVec2 pt) {
        ImVec2 disp = {pt.x - center.x, pt.y - center.y};
        double d2 = disp.x*disp.x + disp.y*disp.y;
        return d2 < radius*radius;
    }
};

struct RectPicker {
    ImVec2 tl;
    ImVec2 br;

    bool contains(ImVec2 pt) {
        return (tl.x <= pt.x && tl.y <= pt.y &&
                br.x >= pt.x && br.y >= pt.y);
    }
};

struct ImGuiOverlayable {
    float data_width;
    float data_height;
    float display_width;
    float display_height;
    float corner_x;
    float corner_y;
    float scale;
    float data_center_x;
    float data_center_y;

    static ImGuiOverlayable Image(void* texture_id, float width, float height, float display_width = 0) {
        if (display_width == 0) {
            display_width = width;
        }
        ImGuiOverlayable result;
        result.scale = display_width / width;
        result.display_width = display_width;
        result.display_height = result.scale* height;
        const auto screen_pos = ImGui::GetCursorScreenPos();
        result.corner_x = screen_pos.x;
        result.corner_y = screen_pos.y;
        result.data_center_x = width/2;
        result.data_center_y = height/2;
        ImGui::Image(texture_id, ImVec2{result.display_width, result.display_height});
        return result;
    };

    static ImGuiOverlayable Rectangle(float width, float height, float display_width = 0) {
        if (display_width == 0) {
            display_width = width;
        }
        ImGuiOverlayable result;
        result.scale = display_width / width;
        result.display_width = display_width;
        result.display_height = result.scale * height;
        const auto screen_pos = ImGui::GetCursorScreenPos();
        result.corner_x = screen_pos.x;
        result.corner_y = screen_pos.y;
        result.data_center_x = width/2;
        result.data_center_y = height/2;
        
        ImGui::InvisibleButton("rectangle", ImVec2{result.display_width, result.display_height});

        // draw a border around the box
        result.add_rect(0,0, width, height, ImColor(1.0f, 1.0f, 1.0f, 1.0f));
        return result;
    };

    void to_window_coords(float* x, float* y) {
        *x = scale * (*x - data_center_x) + corner_x + display_width/2;
        *y = scale * (*y - data_center_y) + corner_y + display_height/2;
    }

    CirclePicker add_circle(float x, float y, float r, uint32_t color = ImColor(0.0f, 0.0f, 0.0f, 1.0f), float thickness = 1.0) {
        to_window_coords(&x, &y);
        r *= scale;
        ImGui::GetWindowDrawList()->AddCircle(ImVec2{x,y}, r, color, /*num_segments=*/16, thickness);
        return CirclePicker { ImVec2{x,y}, r };
    }

    CirclePicker add_circle_filled(float x, float y, float r, uint32_t color = ImColor(0.0f, 0.0f, 0.0f, 1.0f)) {
        to_window_coords(&x, &y);
        r *= scale;
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2{x,y}, r, color, /*num_segments=*/16);
        return CirclePicker { ImVec2{x,y}, r };
    }
    

    void add_line(float x0, float y0, float x1, float y1, uint32_t color, float thickness = 1.0) {
        to_window_coords(&x0, &y0);
        to_window_coords(&x1, &y1);
        ImGui::GetWindowDrawList()->AddLine(ImVec2{x0, y0}, ImVec2{x1, y1}, color, thickness);
    }

    RectPicker add_rect(float x0, float y0, float x1, float y1, uint32_t color, float thickness = 1.0) {
        to_window_coords(&x0, &y0);
        to_window_coords(&x1, &y1);
        ImGui::GetWindowDrawList()->AddRect(ImVec2{x0, y0}, ImVec2{x1, y1}, color, 0, 0, thickness);
        return RectPicker { ImVec2 {x0,y0}, ImVec2{x1,y1} };
    }

    RectPicker add_rect_filled(float x0, float y0, float x1, float y1, uint32_t color) {
        to_window_coords(&x0, &y0);
        to_window_coords(&x1, &y1);
        ImGui::GetWindowDrawList()->AddRectFilled(ImVec2{x0, y0}, ImVec2{x1, y1}, color, 0, 0);
        return RectPicker { ImVec2 {x0,y0}, ImVec2{x1,y1} };
    }

    void add_text(float x, float y, uint32_t color, const char* text) {
        to_window_coords(&x,&y);
        ImGui::GetWindowDrawList()->AddText(ImVec2{x,y},color,text);
    }
};
