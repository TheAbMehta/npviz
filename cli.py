
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="npviz", description="Neuroplastic Architecture Visualizer")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="Launch the interactive viewer")
    serve_parser.add_argument("log_dir", type=str, help="Path to the recorded run directory")
    serve_parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server")
    serve_parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")

    summary_parser = sub.add_parser("summary", help="Print a text summary of rewiring events")
    summary_parser.add_argument("log_dir", type=str, help="Path to the recorded run directory")

    export_parser = sub.add_parser("export", help="Export static figures (SVG/PDF/PNG)")
    export_parser.add_argument("log_dir", type=str, help="Path to the recorded run directory")
    export_parser.add_argument("-o", "--output", type=str, default="figures/", help="Output directory")
    export_parser.add_argument("-f", "--format", type=str, default="svg", choices=["svg", "pdf", "png"],
                               help="Output format (default: svg)")
    export_parser.add_argument("--step", type=int, default=None,
                               help="Step for step-dependent plots (default: last)")
    export_parser.add_argument("--no-paper-style", action="store_true",
                               help="Disable paper-ready styling")

    video_parser = sub.add_parser("video", help="Generate Manim animation video")
    video_parser.add_argument("log_dir", type=str, help="Path to the recorded run directory")
    video_parser.add_argument("-o", "--output", type=str, default="evolution.mp4", help="Output video path")
    video_parser.add_argument("--scene", type=str, default="overview",
                              choices=["architecture", "importance", "overview"],
                              help="Scene to render (default: overview)")
    video_parser.add_argument("--quality", type=str, default="medium",
                              choices=["low", "medium", "high", "4k"],
                              help="Render quality (default: medium)")

    args = parser.parse_args()

    if args.command == "serve":
        from .viewer import serve
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            print(f"Error: {log_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"Loading data from {log_dir}...")
        serve(log_dir, port=args.port, debug=args.debug)

    elif args.command == "summary":
        from .recorder import Recorder
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            print(f"Error: {log_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        tl = Recorder.load(log_dir)
        print(f"Run: {log_dir.name}")
        print(f"Snapshots: {len(tl.snapshots)}")
        print(f"Events: {len(tl.events)}")
        print(f"Metrics: {len(tl.metrics)}")
        print()

        if tl.snapshots:
            first, last = tl.snapshots[0], tl.snapshots[-1]
            print(f"Initial: {first.n_layers} layers, {first.total_params:,} params, heads={first.heads_per_layer}")
            print(f"Final:   {last.n_layers} layers, {last.total_params:,} params, heads={last.heads_per_layer}")
            print()

        print("Rewiring Events:")
        for e in tl.events:
            loc = f"L{e.layer_idx}" + (f"/H{e.head_idx}" if e.head_idx is not None else "")
            delta = ""
            if e.loss_after is not None:
                d = e.loss_after - e.loss_before
                delta = f" (loss {d:+.4f})"
            print(f"  Step {e.step:>6}: {e.event_type:<14} {loc:<8} {e.reason}{delta}")

    elif args.command == "export":
        from .export import export_run
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            print(f"Error: {log_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"Exporting {args.format.upper()} figures from {log_dir}...")
        paths = export_run(
            log_dir, args.output,
            fmt=args.format,
            step=args.step,
            paper_style=not args.no_paper_style,
        )
        for p in paths:
            print(f"  {p}")
        print(f"Done. {len(paths)} files written to {args.output}")

    elif args.command == "video":
        from .video import render_video
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            print(f"Error: {log_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"Rendering {args.scene} scene ({args.quality} quality)...")
        output = render_video(
            log_dir, args.output,
            scene=args.scene,
            quality=args.quality,
        )
        print(f"Done. Video saved to {output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
