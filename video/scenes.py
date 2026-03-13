
from __future__ import annotations

try:
    from manim import (
        BLACK,
        DOWN,
        LEFT,
        RIGHT,
        UP,
        Circle,
        Create,
        DashedLine,
        FadeIn,
        FadeOut,
        GrowFromCenter,
        Line,
        Rectangle,
        Scene,
        ShrinkToCenter,
        Text,
        VGroup,
        Write,
        color as manim_color,
    )

    _HAS_MANIM = True
except ImportError:
    _HAS_MANIM = False

from ..schema import Timeline
from .styles import DARK_THEME, EVENT_ANIMATIONS, importance_to_color


def _check_manim():
    if not _HAS_MANIM:
        raise ImportError("manim is required for video generation. Install with: pip install npviz[video]")


class ArchitectureEvolution(Scene):

    def __init__(self, timeline: Timeline, theme: dict | None = None, **kwargs):
        _check_manim()
        super().__init__(**kwargs)
        self.timeline = timeline
        self.theme = theme or DARK_THEME

    def construct(self):
        bg_color = self.theme["background"]
        self.camera.background_color = bg_color

        title = Text("Architecture Evolution", font_size=48, color=self.theme["text"])
        subtitle = Text(
            f"{len(self.timeline.events)} rewiring events",
            font_size=24,
            color=self.theme["foreground"],
        )
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        snap = self.timeline.snapshots[0] if self.timeline.snapshots else None
        if snap is None:
            return

        layer_groups, head_circles, connections = self._build_topology(snap)
        all_elements = VGroup(*layer_groups, *connections)
        all_elements.move_to([0, 0, 0])

        self.play(FadeIn(all_elements))
        self.wait(0.5)

        step_text = Text(f"Step: 0", font_size=20, color=self.theme["text"])
        step_text.to_corner(UP + RIGHT, buff=0.5)
        self.play(FadeIn(step_text))

        for event in self.timeline.events:
            anim_info = EVENT_ANIMATIONS.get(event.event_type, {"color": "#ffffff", "effect": "fade"})
            color = anim_info["color"]

            new_step_text = Text(f"Step: {event.step}", font_size=20, color=self.theme["text"])
            new_step_text.to_corner(UP + RIGHT, buff=0.5)

            label = Text(
                f"{event.event_type.replace('_', ' ')} L{event.layer_idx}"
                + (f"/H{event.head_idx}" if event.head_idx is not None else ""),
                font_size=18,
                color=color,
            )
            label.to_edge(DOWN, buff=0.5)

            self.play(
                FadeOut(step_text),
                FadeIn(new_step_text),
                FadeIn(label),
                run_time=0.3,
            )
            step_text = new_step_text

            if event.event_type == "prune_head" and event.head_idx is not None:
                li = event.layer_idx
                hi = event.head_idx
                if li < len(head_circles) and hi < len(head_circles[li]):
                    target = head_circles[li][hi]
                    target.set_color(color)
                    self.play(ShrinkToCenter(target), run_time=0.5)

            elif event.event_type == "grow_head":
                li = event.layer_idx
                if li < len(head_circles):
                    new_circle = Circle(radius=0.15, color=color, fill_opacity=0.8)
                    if head_circles[li]:
                        new_circle.next_to(head_circles[li][-1], RIGHT, buff=0.1)
                    else:
                        new_circle.move_to(layer_groups[li].get_center())
                    head_circles[li].append(new_circle)
                    self.play(GrowFromCenter(new_circle), run_time=0.5)

            elif event.event_type == "reconnect":
                li = event.layer_idx
                if li < len(layer_groups) and li + 3 < len(layer_groups):
                    start = layer_groups[li].get_center()
                    end = layer_groups[min(li + 3, len(layer_groups) - 1)].get_center()
                    dash = DashedLine(start, end, color=color, stroke_width=2)
                    connections.append(dash)
                    self.play(Create(dash), run_time=0.5)

            elif event.event_type == "prune_layer":
                li = event.layer_idx
                if li < len(layer_groups):
                    self.play(FadeOut(layer_groups[li]), run_time=0.5)

            self.wait(0.3)
            self.play(FadeOut(label), run_time=0.2)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def _build_topology(self, snap):
        layer_groups = []
        head_circles = []
        connections = []
        max_heads = max(snap.heads_per_layer) if snap.heads_per_layer else 1
        spacing_y = 0.8
        spacing_x = 0.35

        for li, n_heads in enumerate(snap.heads_per_layer):
            circles = []
            for hi in range(n_heads):
                c = Circle(
                    radius=0.15,
                    color=self.theme["layer_stroke"],
                    fill_color=self.theme["layer_fill"],
                    fill_opacity=0.8,
                )
                x = (hi - (n_heads - 1) / 2) * spacing_x
                y = -li * spacing_y
                c.move_to([x, y, 0])
                circles.append(c)
            group = VGroup(*circles)
            layer_groups.append(group)
            head_circles.append(circles)

        for li in range(len(snap.heads_per_layer) - 1):
            line = Line(
                layer_groups[li].get_center() + [0, -0.2, 0],
                layer_groups[li + 1].get_center() + [0, 0.2, 0],
                color=self.theme["connection"],
                stroke_width=1,
            )
            connections.append(line)

        return layer_groups, head_circles, connections


class ImportanceEvolution(Scene):

    def __init__(self, timeline: Timeline, theme: dict | None = None, **kwargs):
        _check_manim()
        super().__init__(**kwargs)
        self.timeline = timeline
        self.theme = theme or DARK_THEME

    def construct(self):
        self.camera.background_color = self.theme["background"]

        title = Text("Importance Evolution", font_size=48, color=self.theme["text"])
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        if not self.timeline.importance:
            return

        first_imp = self.timeline.importance[0]
        n_layers = len(first_imp.scores)
        max_heads = max(len(row) for row in first_imp.scores)

        cell_size = 0.4
        grid: list[list[Rectangle]] = []

        for li in range(n_layers):
            row = []
            for hi in range(max_heads):
                if hi < len(first_imp.scores[li]):
                    score = first_imp.scores[li][hi]
                    color = importance_to_color(score)
                else:
                    color = "#000000"

                rect = Rectangle(
                    width=cell_size, height=cell_size,
                    fill_color=color, fill_opacity=1.0,
                    stroke_color=self.theme["grid"], stroke_width=0.5,
                )
                x = (hi - (max_heads - 1) / 2) * (cell_size + 0.05)
                y = -(li - (n_layers - 1) / 2) * (cell_size + 0.05)
                rect.move_to([x, y, 0])
                row.append(rect)
            grid.append(row)

        all_rects = VGroup(*[r for row in grid for r in row])
        self.play(FadeIn(all_rects))
        self.wait(0.5)

        step_text = Text(f"Step: {first_imp.step}", font_size=20, color=self.theme["text"])
        step_text.to_corner(UP + RIGHT, buff=0.5)
        self.play(FadeIn(step_text))

        for imp in self.timeline.importance[1:]:
            new_step = Text(f"Step: {imp.step}", font_size=20, color=self.theme["text"])
            new_step.to_corner(UP + RIGHT, buff=0.5)

            animations = [FadeOut(step_text), FadeIn(new_step)]
            for li in range(min(n_layers, len(imp.scores))):
                for hi in range(min(max_heads, len(imp.scores[li]))):
                    score = imp.scores[li][hi]
                    new_color = importance_to_color(score)
                    grid[li][hi].generate_target()
                    grid[li][hi].target.set_fill(color=new_color, opacity=1.0)

            self.play(*animations, run_time=0.5)
            for li in range(min(n_layers, len(imp.scores))):
                for hi in range(min(max_heads, len(imp.scores[li]))):
                    score = imp.scores[li][hi]
                    grid[li][hi].set_fill(color=importance_to_color(score), opacity=1.0)

            step_text = new_step
            self.wait(0.3)

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class TrainingOverview(Scene):

    def __init__(self, timeline: Timeline, theme: dict | None = None, **kwargs):
        _check_manim()
        super().__init__(**kwargs)
        self.timeline = timeline
        self.theme = theme or DARK_THEME

    def construct(self):
        self.camera.background_color = self.theme["background"]

        title = Text("Training Overview", font_size=48, color=self.theme["text"])
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        if not self.timeline.metrics:
            return

        metrics = sorted(self.timeline.metrics, key=lambda m: m.step)
        max_step = metrics[-1].step
        max_loss = max(m.loss for m in metrics)

        chart_width = 5.0
        chart_height = 3.0
        chart_origin = [-3.5, -1.5, 0]

        x_axis = Line(
            chart_origin,
            [chart_origin[0] + chart_width, chart_origin[1], 0],
            color=self.theme["foreground"],
            stroke_width=1,
        )
        y_axis = Line(
            chart_origin,
            [chart_origin[0], chart_origin[1] + chart_height, 0],
            color=self.theme["foreground"],
            stroke_width=1,
        )
        x_label = Text("Step", font_size=14, color=self.theme["text"])
        x_label.next_to(x_axis, DOWN, buff=0.2)
        y_label = Text("Loss", font_size=14, color=self.theme["text"])
        y_label.next_to(y_axis, LEFT, buff=0.2)

        self.play(Create(x_axis), Create(y_axis), FadeIn(x_label), FadeIn(y_label))

        prev_point = None
        event_idx = 0
        event_steps = {e.step for e in self.timeline.events}

        mini_label = Text("Architecture", font_size=18, color=self.theme["text"])
        mini_label.move_to([3.5, 2, 0])
        self.play(FadeIn(mini_label))

        segment_size = max(1, len(metrics) // 40)
        lines = []
        for i, m in enumerate(metrics):
            x = chart_origin[0] + (m.step / max_step) * chart_width
            y = chart_origin[1] + (m.loss / max_loss) * chart_height
            point = [x, y, 0]

            if prev_point is not None and i % segment_size == 0:
                line = Line(prev_point, point, color="#2c3e50", stroke_width=2)
                lines.append(line)
                self.add(line)

            if m.step in event_steps:
                vline = DashedLine(
                    [x, chart_origin[1], 0],
                    [x, chart_origin[1] + chart_height, 0],
                    color=EVENT_ANIMATIONS.get(
                        next(e.event_type for e in self.timeline.events if e.step == m.step),
                        {},
                    ).get("color", "#ffffff"),
                    stroke_width=1,
                )
                self.play(Create(vline), run_time=0.2)

            prev_point = point

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


SCENE_REGISTRY = {
    "architecture": ArchitectureEvolution,
    "importance": ImportanceEvolution,
    "overview": TrainingOverview,
}
