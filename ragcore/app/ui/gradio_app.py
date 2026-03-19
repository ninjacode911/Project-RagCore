import json
import logging

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:7860"

CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: #6b7280;
    font-size: 1rem;
    margin: 0;
}
.stat-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.answer-box {
    border-left: 4px solid #667eea;
    padding-left: 1rem;
    margin-top: 0.5rem;
}
.source-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.75rem;
    margin: 0.5rem 0;
}
.upload-zone {
    border: 2px dashed #667eea !important;
    border-radius: 12px !important;
    background: #f8f9ff !important;
}
.search-bar textarea {
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    border: 2px solid #e5e7eb !important;
    padding: 12px 16px !important;
}
.search-bar textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
}
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 10px 24px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}
.danger-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    border: none !important;
    border-radius: 10px !important;
}
.filter-row {
    background: #f9fafb;
    border-radius: 10px;
    padding: 8px 12px;
}
.doc-table {
    border-radius: 10px !important;
    overflow: hidden !important;
}
footer { display: none !important; }
.tab-nav button {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}
.tab-nav button.selected {
    border-bottom: 3px solid #667eea !important;
    color: #667eea !important;
}
"""


def upload_document(file):
    if file is None:
        return "Please select a file to upload."
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name.split("/")[-1].split("\\")[-1], f)}
            response = httpx.post(f"{API_BASE}/api/ingest", files=files, timeout=120)
        if response.status_code == 200:
            data = response.json()
            return (
                f"### Document Uploaded\n\n"
                f"| Detail | Value |\n"
                f"|--------|-------|\n"
                f"| **File** | {data['filename']} |\n"
                f"| **Chunks** | {data['num_chunks']} |\n"
                f"| **ID** | `{data['document_id'][:12]}...` |\n"
            )
        else:
            detail = response.json().get("detail", response.text)
            return f"**Upload failed:** {detail}"
    except Exception as e:
        return f"**Upload failed:** {e}"


def _fetch_documents():
    """Fetch documents from API. Returns list of doc dicts or empty list."""
    try:
        response = httpx.get(f"{API_BASE}/api/documents", timeout=30)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception:
        pass
    return []


def list_documents():
    docs = _fetch_documents()
    if not docs:
        return [["—", "—", "—", "—"]]
    return [
        [
            d.get("source", ""),
            d.get("doc_type", "").upper(),
            str(d.get("num_chunks", 0)),
            d.get("document_id", ""),
        ]
        for d in docs
    ]


def get_delete_choices():
    """Build dropdown choices: 'filename (doc_id)' for each indexed document."""
    docs = _fetch_documents()
    if not docs:
        return gr.update(choices=[], value=None)
    choices = [
        f"{d.get('source', 'unknown')}  [{d.get('document_id', '')}]"
        for d in docs
    ]
    return gr.update(choices=choices, value=None)


def get_doc_count():
    try:
        response = httpx.get(f"{API_BASE}/api/documents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            total = data.get("total", 0)
            docs = data.get("documents", [])
            total_chunks = sum(d.get("num_chunks", 0) for d in docs)
            return f"**{total}** documents | **{total_chunks}** chunks indexed"
        return "Unable to fetch stats"
    except Exception:
        return "Connecting..."


def delete_document(selection):
    if not selection:
        return "Select a document to delete."
    # Extract document ID from "filename  [doc_id]" format
    if "[" in selection and selection.endswith("]"):
        doc_id = selection.rsplit("[", 1)[1][:-1]
    else:
        doc_id = selection.strip()
    try:
        response = httpx.delete(f"{API_BASE}/api/documents/{doc_id}", timeout=30)
        if response.status_code == 200:
            filename = selection.split("  [")[0] if "  [" in selection else doc_id[:12]
            return f"**'{filename}'** deleted successfully."
        detail = response.json().get("detail", response.text)
        return f"**Error:** {detail}"
    except Exception as e:
        return f"**Delete failed:** {e}"


def ask_question(query, doc_type_filter, stream_mode):
    if not query or not query.strip():
        yield "Please enter a question."
        return

    payload = {
        "query": query.strip(),
        "top_k": 10,
        "rerank_top_k": 5,
        "stream": stream_mode,
    }

    if doc_type_filter and doc_type_filter != "All":
        payload["filters"] = {"doc_type": doc_type_filter.lower()}

    try:
        if stream_mode:
            with httpx.stream(
                "POST",
                f"{API_BASE}/api/ask",
                json=payload,
                timeout=120,
            ) as response:
                answer = ""
                sources_text = ""
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "text" in data:
                            answer += data["text"]
                            yield (
                                f"<div class='answer-box'>\n\n{answer}\n\n</div>\n\n"
                                f"<sub>Generating...</sub>"
                            )
                        if data.get("done"):
                            sources = data.get("sources", [])
                            sources_text = _format_sources(sources)
                            time_ms = data.get("time_ms", 0)
                            model = data.get("model", "")
                            footer = f"\n\n<sub>{model} | {time_ms:.0f}ms</sub>"
                            yield (
                                f"<div class='answer-box'>\n\n{answer}\n\n</div>"
                                f"{sources_text}{footer}"
                            )
        else:
            response = httpx.post(
                f"{API_BASE}/api/ask",
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer generated.")
                sources = data.get("sources", [])
                sources_text = _format_sources_full(sources)
                time_ms = data.get("generation_time_ms", 0)
                model = data.get("model", "")
                footer = f"\n\n<sub>{model} | {time_ms:.0f}ms</sub>"
                yield (
                    f"<div class='answer-box'>\n\n{answer}\n\n</div>"
                    f"{sources_text}{footer}"
                )
            else:
                yield f"**Error:** {response.text}"
    except Exception as e:
        yield f"**Error:** {e}"


def _format_sources(sources):
    if not sources:
        return ""
    text = "\n\n---\n#### Sources\n\n"
    for i, s in enumerate(sources, 1):
        source_name = s.get("source", "unknown")
        score = s.get("score", 0)
        snippet = s.get("text", "")[:120].replace("\n", " ")
        text += (
            f"<div class='source-card'>\n\n"
            f"**[{i}]** `{source_name}` — relevance: {score:.3f}\n\n"
            f"> {snippet}...\n\n"
            f"</div>\n\n"
        )
    return text


def _format_sources_full(sources):
    if not sources:
        return ""
    text = "\n\n---\n#### Sources\n\n"
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        source_name = meta.get("source", "unknown")
        score = s.get("score", 0)
        snippet = s.get("text", "")[:120].replace("\n", " ")
        text += (
            f"<div class='source-card'>\n\n"
            f"**[{i}]** `{source_name}` — relevance: {score:.3f}\n\n"
            f"> {snippet}...\n\n"
            f"</div>\n\n"
        )
    return text


def create_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="RagCore — Smart Document Q&A") as demo:

        # Inject CSS via style tag since Gradio 6.x doesn't accept css in Blocks()
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # Header
        gr.HTML(
            """
            <div class="main-header">
                <h1>RagCore</h1>
                <p>Smart Document Q&A — Hybrid Search + Gemini Flash</p>
            </div>
            """
        )

        # Stats bar
        stats_display = gr.Markdown(value="Connecting...", elem_classes=["stat-card"])
        demo.load(fn=get_doc_count, outputs=stats_display)

        with gr.Tab("Ask", elem_id="ask-tab"):
            gr.Markdown("#### Ask your documents anything")

            with gr.Group():
                query_input = gr.Textbox(
                    placeholder="e.g. What are the key findings? / Summarize the report / Compare approaches...",
                    lines=2,
                    show_label=False,
                    elem_classes=["search-bar"],
                    container=False,
                )

            with gr.Row(elem_classes=["filter-row"]):
                doc_type_filter = gr.Dropdown(
                    choices=["All", "PDF", "TXT", "HTML"],
                    value="All",
                    label="Document Type",
                    scale=1,
                    min_width=120,
                )
                stream_toggle = gr.Checkbox(
                    label="Stream response",
                    value=True,
                    scale=1,
                )
                ask_btn = gr.Button(
                    "Ask",
                    variant="primary",
                    scale=1,
                    min_width=120,
                    elem_classes=["primary-btn"],
                )

            answer_output = gr.Markdown(
                value="*Upload a document and ask a question to get started.*",
            )

            ask_btn.click(
                fn=ask_question,
                inputs=[query_input, doc_type_filter, stream_toggle],
                outputs=answer_output,
            )
            query_input.submit(
                fn=ask_question,
                inputs=[query_input, doc_type_filter, stream_toggle],
                outputs=answer_output,
            )

            gr.Markdown("#### Try these examples")
            gr.Examples(
                examples=[
                    ["What are the key points in the uploaded documents?"],
                    ["Summarize all documents"],
                    ["Compare the main topics across all documents"],
                    ["List the most important findings"],
                ],
                inputs=query_input,
            )

        with gr.Tab("Documents", elem_id="docs-tab"):
            gr.Markdown("#### Upload & Manage Documents")

            with gr.Row():
                with gr.Column(scale=3):
                    file_upload = gr.File(
                        label="Drop your file here",
                        file_types=[".pdf", ".txt", ".html", ".htm"],
                        elem_classes=["upload-zone"],
                    )
                with gr.Column(scale=1, min_width=160):
                    upload_btn = gr.Button(
                        "Upload & Index",
                        variant="primary",
                        elem_classes=["primary-btn"],
                        size="lg",
                    )
                    gr.Markdown(
                        "<sub>Supported: PDF, TXT, HTML</sub>"
                    )

            upload_status = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("#### Indexed Documents")

            doc_table = gr.Dataframe(
                headers=["Filename", "Type", "Chunks", "Document ID"],
                label="",
                interactive=False,
                wrap=True,
                elem_classes=["doc-table"],
            )
            refresh_btn = gr.Button("Refresh", size="sm")

            gr.Markdown("---")
            gr.Markdown("#### Delete a Document")
            with gr.Row():
                delete_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select document to delete",
                    scale=3,
                )
                delete_btn = gr.Button(
                    "Delete",
                    variant="stop",
                    scale=1,
                    elem_classes=["danger-btn"],
                )
            delete_status = gr.Markdown()

            # Refresh: updates table, dropdown, and stats
            def refresh_all():
                return list_documents(), get_delete_choices(), get_doc_count()

            refresh_btn.click(
                fn=refresh_all,
                outputs=[doc_table, delete_dropdown, stats_display],
            )

            # After upload: refresh table, dropdown, and stats
            upload_btn.click(
                fn=upload_document,
                inputs=file_upload,
                outputs=upload_status,
            ).then(
                fn=refresh_all,
                outputs=[doc_table, delete_dropdown, stats_display],
            )

            # Delete: delete then refresh everything
            delete_btn.click(
                fn=delete_document,
                inputs=delete_dropdown,
                outputs=delete_status,
            ).then(
                fn=refresh_all,
                outputs=[doc_table, delete_dropdown, stats_display],
            )

        # Footer
        gr.HTML(
            """
            <div style="text-align:center; padding: 1rem 0 0.5rem 0; color: #9ca3af; font-size: 0.8rem;">
                RagCore v0.1.0
            </div>
            """
        )

    return demo
