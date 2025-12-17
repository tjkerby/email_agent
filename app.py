from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from email_agent import EmailAgent, EmailMessage, EvaluationAndReply
from grading_serialization import grading_result_to_storage
from scenario_loader import load_scenario
from rubric_loader import load_rubric, RubricDefinition

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
DEFAULT_MODEL = "gpt-oss:20b"


def _scenario_label(filename: str) -> str:
    scenario = load_scenario_cached(filename)
    return scenario.name.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def list_scenario_files() -> list[str]:
    if not SCENARIO_DIR.exists():
        return []
    return sorted(path.name for path in SCENARIO_DIR.glob("*.json"))


@st.cache_data(show_spinner=False)
def load_scenario_cached(filename: str):
    return load_scenario(SCENARIO_DIR / filename)


def _rubric_label(filename: str) -> str:
    rubric = load_rubric_cached(filename)
    return rubric.name or filename


def _list_rubric_names() -> list[str]:
    if not RUBRIC_DIR.exists():
        return []
    names: set[str] = set()
    for pattern in ("*.json", "*.yaml", "*.yml"):
        for path in RUBRIC_DIR.glob(pattern):
            names.add(path.name)
    return sorted(names)


@st.cache_data(show_spinner=False)
def list_rubric_files() -> list[str]:
    return _list_rubric_names()


@st.cache_data(show_spinner=False)
def load_rubric_cached(filename: str) -> RubricDefinition:
    return load_rubric(RUBRIC_DIR / filename)


def _ensure_starter_thread(agent: EmailAgent, cache_key: str) -> list[EmailMessage]:
    cache = st.session_state.setdefault("starter_cache", {})
    if cache_key not in cache:
        with st.spinner("Generating starter email..."):
            cache[cache_key] = agent.build_starter_thread()
    return cache[cache_key]


def _reset_starter_thread(cache_key: str) -> None:
    cache = st.session_state.setdefault("starter_cache", {})
    cache.pop(cache_key, None)


def _store_latest_result(cache_key: str, result: EvaluationAndReply, student_body: str) -> None:
    st.session_state["latest_result"] = {
        "cache_key": cache_key,
        "result": result,
        "student_body": student_body,
    }


def _get_latest_result(cache_key: str) -> tuple[EvaluationAndReply, str] | None:
    latest = st.session_state.get("latest_result")
    if latest and latest.get("cache_key") == cache_key:
        return latest["result"], latest["student_body"]
    return None


def _render_grading(
    result: EvaluationAndReply,
    student_body: str,
    subject: str,
    rubric_name: str,
) -> None:
    grading = result.grading

    with st.expander("Grading Results", expanded=True):
        st.caption(f"Rubric: {rubric_name}")
        total_score = f"{grading.total_score}/{grading.max_total_score}"
        st.metric("Total Score", total_score)

        if grading.scores:
            score_rows = [
                {"Dimension": score.name, "Score": f"{score.score}/{score.max_score}"}
                for score in grading.scores
            ]
            st.table(score_rows)

        st.markdown("**Overall Comment**")
        st.write(grading.overall_comment or "(no comment returned)")

        st.markdown("**Revision Example**")
        st.write(grading.revision_example or "(no revision provided)")

        st.subheader("Student Email (for reference)")
        st.caption(f"Subject: Re: {subject}")
        st.code(student_body, language="markdown")

        st.subheader("AI Counterpart Reply")
        st.code(result.counterpart_reply.strip(), language="markdown")

        payload = grading_result_to_storage(grading)
        st.subheader("JSON Stored in Supabase")
        st.code(json.dumps(payload, indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="Email Etiquette Trainer", layout="wide")
    st.title("Email Etiquette Trainer")
    st.caption("Pick a scenario, draft a response, and get instant feedback.")

    scenario_files = list_scenario_files()
    if not scenario_files:
        st.error("No scenario JSON files found in the `scenarios/` directory.")
        st.stop()

    st.sidebar.header("Scenario, Model & Rubric")
    selected_file = st.sidebar.selectbox(
        "Scenario",
        options=scenario_files,
        format_func=_scenario_label,
    )
    scenario = load_scenario_cached(selected_file)

    rubric_files = list_rubric_files()
    if not rubric_files:
        st.error("No rubric definition files found in the `rubrics/` directory.")
        st.stop()

    selected_rubric = st.sidebar.selectbox(
        "Rubric",
        options=rubric_files,
        format_func=_rubric_label,
    )
    rubric_definition = load_rubric_cached(selected_rubric)

    model_name = st.sidebar.text_input("Ollama model", value=DEFAULT_MODEL)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    base_url_raw = st.sidebar.text_input(
        "Ollama base URL (optional)",
        value="",
        placeholder="http://localhost:11434",
    )
    base_url = base_url_raw.strip() or None

    cache_key = f"{selected_file}|{selected_rubric}|{model_name}|{temperature}|{base_url or 'default'}"

    if st.sidebar.button("Regenerate starter email"):
        _reset_starter_thread(cache_key)

    try:
        agent = EmailAgent(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            scenario=scenario,
        )
    except Exception as exc:
        st.error(f"Failed to initialize EmailAgent: {exc}")
        st.stop()

    try:
        starter_thread = _ensure_starter_thread(agent, cache_key)
    except Exception as exc:
        st.error(f"Unable to generate starter email: {exc}")
        st.stop()

    starter_email = starter_thread[0]

    with st.expander("Scenario Overview", expanded=True):
        st.markdown(f"**Description:** {scenario.description}")
        info_cols = st.columns(3)
        info_cols[0].markdown(f"**Environment:** {scenario.environment}")
        info_cols[1].markdown(f"**Counterpart Role:** {scenario.counterpart_role}")
        info_cols[2].markdown(f"**Grading Focus:** {scenario.grading_focus or '—'}")

        st.markdown("**What the student must do:**")
        st.write(scenario.student_task)

        if scenario.counterpart_style:
            st.markdown("**Counterpart tone:**")
            st.write(scenario.counterpart_style)

    with st.expander("Rubric Overview", expanded=True):
        st.markdown(f"**{rubric_definition.name}**")
        if rubric_definition.description:
            st.write(rubric_definition.description)
        rubric_rows = [
            {
                "Dimension": item.name,
                "Max Score": item.max_score,
                "Description": item.description,
            }
            for item in rubric_definition.items
        ]
        st.table(rubric_rows)

    st.divider()

    with st.expander("Starter Email (Manager → Student)", expanded=True):
        st.markdown(f"**Subject:** {starter_email.subject}")
        st.markdown(f"**From:** {starter_email.sender}")
        st.code(starter_email.body.strip(), language="markdown")

    with st.expander("Your Turn — Student Reply", expanded=True):
        placeholder = "Hello ..."
        student_body = st.text_area(
            "Draft your email",
            value=st.session_state.get("draft_email", ""),
            placeholder=placeholder,
            height=220,
        )
    st.session_state["draft_email"] = student_body

    if st.button("Grade my email", type="primary"):
        if not student_body.strip():
            st.warning("Please enter an email before grading.")
        else:
            student_email = EmailMessage(
                sender="Student",
                subject=f"Re: {starter_email.subject}",
                body=student_body.strip(),
            )
            with st.spinner("Scoring email and drafting counterpart reply..."):
                try:
                    result = agent.evaluate_and_respond(
                        prior_thread=starter_thread,
                        student_email=student_email,
                        rubric=rubric_definition.items,
                    )
                except Exception as exc:
                    st.error(f"Failed to run evaluation: {exc}")
                else:
                    _store_latest_result(cache_key, result, student_email.body)

    latest = _get_latest_result(cache_key)
    if latest:
        result, body = latest
        _render_grading(result, body, starter_email.subject, rubric_definition.name)


if __name__ == "__main__":
    main()
