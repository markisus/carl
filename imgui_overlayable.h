struct ImGuiOverlayable {
    float data_width;
    float data_height;
    float display_width;
    float display_height;
    float corner_x;
    float corner_y;
    float scale;

    static ImGuiOverlayable Image(void* texture_id, float width, float height, float display_width = 0) {
        if (display_width == 0) {
            display_width = width;
        }
        ImGuiOverlayable result;
        result.scale = display_width/ width;
        result.display_width = display_width;
        result.display_height = result.scale* height;
        const auto screen_pos = ImGui::GetCursorScreenPos();
        result.corner_x = screen_pos.x;
        result.corner_y = screen_pos.y;
        ImGui::Image(texture_id, ImVec2{result.display_width, result.display_height});
        return result;
    };

    void to_window_coords(float* x, float* y) {
        *x = scale * *x + corner_x;
        *y = scale * *y + corner_y;
    }

    void add_circle(float x, float y, float r, uint32_t color = ImColor(0.0f, 0.0f, 0.0f, 1.0f), float thickness = 1.0) {
        to_window_coords(&x, &y);
        r *= scale;
        ImGui::GetWindowDrawList()->AddCircle(ImVec2{x,y}, r, color, /*num_segments=*/16, thickness);
    }

    void add_line(float x0, float y0, float x1, float y1, uint32_t color, float thickness = 1.0) {
        to_window_coords(&x0, &y0);
        to_window_coords(&x1, &y1);
        ImGui::GetWindowDrawList()->AddLine(ImVec2{x0, y0}, ImVec2{x1, y1}, color, thickness);
    }
};
