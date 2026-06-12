"""LangGraph 더빙 워크플로우 구성."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from dubbing_agent.graph.nodes import (
    build_context_node,
    compose_audio_node,
    duration_adjust_node,
    extract_audio_node,
    final_quality_gate_node,
    generate_subtitle_node,
    global_translation_review_node,
    inspect_source_node,
    load_source_subtitle_node,
    mux_video_node,
    next_segment_node,
    prepare_video_node,
    stt_node,
    translation_qc_node,
    translate_segment_node,
    tts_duration_check_node,
    tts_node,
    write_quality_report_node,
)
from dubbing_agent.graph.routes import (
    route_duration_adjust,
    route_has_segments,
    route_loaded_source_subtitle,
    route_global_translation_review,
    route_next_segment,
    route_source_text,
    route_translation_qc,
    route_tts_duration,
)
from dubbing_agent.state import DubbingState


def build_dubbing_graph(checkpoint: bool = False):
    """더빙 워크플로우 그래프를 컴파일한다."""
    graph = StateGraph(DubbingState)

    graph.add_node("inspect_source", inspect_source_node)
    graph.add_node("prepare_video", prepare_video_node)
    graph.add_node("load_source_subtitle", load_source_subtitle_node)
    graph.add_node("extract_audio", extract_audio_node)
    graph.add_node("stt", stt_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("translate_segment", translate_segment_node)
    graph.add_node("translation_qc", translation_qc_node)
    graph.add_node("duration_adjust", duration_adjust_node)
    graph.add_node("tts", tts_node)
    graph.add_node("tts_duration_check", tts_duration_check_node)
    graph.add_node("next_segment", next_segment_node)
    graph.add_node("global_translation_review", global_translation_review_node)
    graph.add_node("final_quality_gate", final_quality_gate_node)
    graph.add_node("generate_subtitle", generate_subtitle_node)
    graph.add_node("compose_audio", compose_audio_node)
    graph.add_node("write_quality_report", write_quality_report_node)
    graph.add_node("mux_video", mux_video_node)

    graph.add_edge(START, "inspect_source")
    graph.add_edge("inspect_source", "prepare_video")
    graph.add_conditional_edges(
        "prepare_video",
        route_source_text,
        {
            "load_source_subtitle": "load_source_subtitle",
            "extract_audio": "extract_audio",
        },
    )
    graph.add_conditional_edges(
        "load_source_subtitle",
        route_loaded_source_subtitle,
        {
            "build_context": "build_context",
            "extract_audio": "extract_audio",
        },
    )
    graph.add_edge("extract_audio", "stt")
    graph.add_edge("stt", "build_context")
    graph.add_conditional_edges(
        "build_context",
        route_has_segments,
        {
            "translate_segment": "translate_segment",
            "final_quality_gate": "final_quality_gate",
        },
    )
    graph.add_edge("translate_segment", "translation_qc")
    graph.add_conditional_edges(
        "translation_qc",
        route_translation_qc,
        {
            "translate_segment": "translate_segment",
            "duration_adjust": "duration_adjust",
        },
    )
    graph.add_conditional_edges(
        "duration_adjust",
        route_duration_adjust,
        {
            "translate_segment": "translate_segment",
            "tts": "tts",
            "next_segment": "next_segment",
        },
    )
    graph.add_edge("tts", "tts_duration_check")
    graph.add_conditional_edges(
        "tts_duration_check",
        route_tts_duration,
        {
            "duration_adjust": "duration_adjust",
            "translate_segment": "translate_segment",
            "next_segment": "next_segment",
        },
    )
    graph.add_conditional_edges(
        "next_segment",
        route_next_segment,
        {
            "translate_segment": "translate_segment",
            "translation_qc": "translation_qc",
            "global_translation_review": "global_translation_review",
        },
    )
    graph.add_conditional_edges(
        "global_translation_review",
        route_global_translation_review,
        {
            "translation_qc": "translation_qc",
            "final_quality_gate": "final_quality_gate",
        },
    )
    graph.add_edge("final_quality_gate", "generate_subtitle")
    graph.add_edge("generate_subtitle", "compose_audio")
    graph.add_edge("compose_audio", "mux_video")
    graph.add_edge("mux_video", "write_quality_report")
    graph.add_edge("write_quality_report", END)

    if not checkpoint:
        return graph.compile()

    from langgraph.checkpoint.memory import InMemorySaver

    return graph.compile(checkpointer=InMemorySaver())
