"""
AI-Powered Compliance Checker
Streamlit single-page app with 3 tabs:
  1. Manage Rules  — create/edit/delete compliance profiles
  2. Run Audit     — upload documents and check them
  3. Results       — view pass/fail matrix with evidence
"""

import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.document_parser import extract_text
from core.compliance_engine import (
    check_document, ProviderConfig,
    PROVIDERS, OLLAMA_SUGGESTED_MODELS, get_ollama_models,
)
from core.rule_profiles import list_profiles, load_profile, save_profile, delete_profile

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Compliance Checker",
    page_icon="✅",
    layout="wide",
)

st.title("✅ AI Compliance Checker")
st.caption("Upload any document and check it against plain-English compliance rules — powered by your choice of AI.")

# ── Sidebar: Provider selection ───────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 AI Provider")

    provider_name = st.selectbox(
        "Choose AI provider",
        list(PROVIDERS.keys()),
        help="All providers use the same interface. Ollama is free and runs locally.",
    )

    provider_info = PROVIDERS[provider_name]
    st.caption(f"💰 {provider_info['cost']}")
    st.caption(f"ℹ️ {provider_info['help']}")

    is_ollama = provider_name.startswith("Ollama")

    # Model selection
    if is_ollama:
        ollama_models = get_ollama_models()
        ollama_running = ollama_models != OLLAMA_SUGGESTED_MODELS

        if ollama_running:
            st.success(f"Ollama running — {len(ollama_models)} model(s) found ✓")
        else:
            st.warning("Ollama not detected. Showing default model list.\nStart Ollama and re-open the sidebar to refresh.")

        model = st.selectbox(
            "Ollama model",
            ollama_models,
            help="Run 'ollama pull <model>' in your terminal to download more models.",
        )
        api_key = ""
        st.info("No API key needed for Ollama.")
    else:
        # Map provider to env var name
        env_var = {
            "xAI Grok": "XAI_API_KEY",
            "MiniMax M3": "MINIMAX_API_KEY",
            "Groq (Free tier)": "GROQ_API_KEY",
            "Google Gemini": "GOOGLE_API_KEY",
        }.get(provider_name, "")

        api_key = st.text_input(
            f"{provider_name} API Key",
            value=os.getenv(env_var, ""),
            type="password",
            help=provider_info["help"],
        )
        model = provider_info["default_model"]

        if api_key:
            st.success("API key loaded ✓")
        else:
            st.warning("Enter your API key to run audits.")

    provider_config = ProviderConfig(
        provider_name=provider_name,
        base_url=provider_info["base_url"],
        model=model,
        api_key=api_key,
    )

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Define rules in plain English\n"
        "2. Upload a PDF or DOCX document\n"
        "3. AI checks each rule against the document\n"
        "4. See pass/fail with evidence quotes"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Manage Rules", "🔍 Run Audit", "📊 Results"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manage Rules
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Compliance Rule Profiles")
    st.markdown(
        "A **profile** is a named set of compliance rules written in plain English. "
        "Each rule is one sentence describing what a document *must* contain."
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Saved Profiles**")
        profiles = list_profiles()
        if profiles:
            selected_profile = st.selectbox("Select a profile to view/edit", profiles)
        else:
            selected_profile = None
            st.info("No profiles yet. Create one on the right.")

        if selected_profile and st.button("🗑 Delete this profile", type="secondary"):
            delete_profile(selected_profile)
            st.success(f"Deleted '{selected_profile}'")
            st.rerun()

    with col_right:
        st.markdown("**Create or Edit a Profile**")

        if selected_profile:
            try:
                existing = load_profile(selected_profile)
                default_name = existing["name"]
                default_desc = existing.get("description", "")
                default_rules = "\n".join(existing["rules"])
            except Exception:
                default_name, default_desc, default_rules = "", "", ""
        else:
            default_name, default_desc, default_rules = "", "", ""

        profile_name = st.text_input("Profile name (no spaces)", value=default_name,
                                     placeholder="e.g. digipen_module_profile")
        profile_desc = st.text_input("Description", value=default_desc,
                                     placeholder="e.g. DigiPen module profile compliance rules")
        rules_text = st.text_area(
            "Rules (one per line)",
            value=default_rules,
            height=300,
            placeholder=(
                "The document must include a module code and title.\n"
                "The document must list learning outcomes with action verbs.\n"
                "The document must include an attendance policy."
            ),
        )

        if st.button("💾 Save Profile", type="primary"):
            if not profile_name.strip():
                st.error("Profile name cannot be empty.")
            else:
                rules = [r.strip() for r in rules_text.splitlines() if r.strip()]
                if not rules:
                    st.error("Add at least one rule.")
                else:
                    save_profile(profile_name.strip(), profile_desc.strip(), rules)
                    st.success(f"Saved profile '{profile_name}' with {len(rules)} rules.")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Run Audit
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Run a Compliance Audit")

    # Show which provider is active
    st.info(f"Using **{provider_name}** · model: `{model}`", icon="🤖")

    profiles = list_profiles()
    if not profiles:
        st.warning("No rule profiles found. Create one in the **Manage Rules** tab first.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        chosen_profile = st.selectbox("Select compliance profile", profiles)
        if chosen_profile:
            profile_data = load_profile(chosen_profile)
            st.caption(profile_data.get("description", ""))
            with st.expander(f"View {len(profile_data['rules'])} rules"):
                for i, rule in enumerate(profile_data["rules"], 1):
                    st.markdown(f"{i}. {rule}")

    with col_b:
        uploaded_files = st.file_uploader(
            "Upload documents (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
        )

    ready = uploaded_files and chosen_profile and (api_key or is_ollama)

    if ready:
        if st.button("🚀 Run Audit", type="primary"):
            profile_data = load_profile(chosen_profile)
            rules = profile_data["rules"]
            all_results = {}

            progress = st.progress(0, text="Starting audit...")
            total = len(uploaded_files) * len(rules)
            done = 0

            for uploaded_file in uploaded_files:
                st.markdown(f"**Checking:** {uploaded_file.name}")
                file_bytes = uploaded_file.read()

                with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                    try:
                        doc_text = extract_text(file_bytes, uploaded_file.name)
                    except Exception as e:
                        st.error(f"Could not parse {uploaded_file.name}: {e}")
                        continue

                results = []
                for rule in rules:
                    progress.progress(done / total, text=f"Checking rule: {rule[:60]}...")
                    rule_results = check_document(doc_text, [rule], provider_config)
                    results.extend(rule_results)
                    done += 1
                    time.sleep(0.05)

                all_results[uploaded_file.name] = results

            progress.progress(1.0, text="Audit complete!")
            st.session_state["audit_results"] = all_results
            st.session_state["audit_profile"] = chosen_profile
            st.session_state["audit_provider"] = provider_name
            st.success("Audit complete! Go to the **Results** tab to view findings.")

    elif uploaded_files and not api_key and not is_ollama:
        st.error("Enter your API key in the sidebar before running the audit.")
    elif not uploaded_files:
        st.info("Upload at least one document to begin.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Audit Results")

    if "audit_results" not in st.session_state:
        st.info("No audit results yet. Run an audit in the **Run Audit** tab.")
        st.stop()

    audit_results: dict = st.session_state["audit_results"]
    audit_profile: str = st.session_state.get("audit_profile", "")
    audit_provider: str = st.session_state.get("audit_provider", "")

    st.caption(f"Profile: **{audit_profile}** · Provider: **{audit_provider}**")

    # ── Summary matrix ────────────────────────────────────────────────────────
    st.markdown("### Summary Matrix")

    matrix_rows = []
    for doc_name, rule_results in audit_results.items():
        passed = sum(1 for r in rule_results if r.result == "pass")
        failed = sum(1 for r in rule_results if r.result == "fail")
        uncertain = sum(1 for r in rule_results if r.result == "uncertain")
        matrix_rows.append({
            "Document": doc_name,
            "✅ Pass": passed,
            "❌ Fail": failed,
            "⚠️ Uncertain": uncertain,
            "Total Rules": len(rule_results),
        })

    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

    # ── Per-document detail ───────────────────────────────────────────────────
    st.markdown("### Detailed Findings")

    for doc_name, rule_results in audit_results.items():
        with st.expander(f"📄 {doc_name}", expanded=True):
            for r in rule_results:
                icon = {"pass": "✅", "fail": "❌", "uncertain": "⚠️"}.get(r.result, "❓")
                confidence_pct = int(r.confidence * 100)

                cols = st.columns([0.05, 0.55, 0.2, 0.2])
                cols[0].markdown(icon)
                cols[1].markdown(f"**{r.rule}**")
                cols[2].caption(f"Confidence: {confidence_pct}%")
                cols[3].caption(r.result.upper())

                if r.evidence and r.evidence != "Not found":
                    st.caption(f"📌 Evidence: *\"{r.evidence}\"*")
                if r.reason:
                    st.caption(f"💬 {r.reason}")
                st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("### Export Results")
    export_rows = []
    for doc_name, rule_results in audit_results.items():
        for r in rule_results:
            export_rows.append({
                "Document": doc_name,
                "Rule": r.rule,
                "Result": r.result,
                "Confidence": r.confidence,
                "Evidence": r.evidence,
                "Reason": r.reason,
            })

    csv = pd.DataFrame(export_rows).to_csv(index=False)
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv,
        file_name="compliance_audit_results.csv",
        mime="text/csv",
    )
