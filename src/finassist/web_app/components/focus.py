"""One-shot focus for the app-owned chat composer, without generated DOM selectors."""

import streamlit as st

# Static application code only: no user text, model output, or external scripts.
# The CSS class comes from the documented key of our own st.container.
_FOCUS_COMPOSER = """
<script>
(() => {
    const focusEditor = () => {
        const editor = document.querySelector('.st-key-chat_composer textarea');
        if (!editor) return false;
        editor.scrollIntoView({behavior: 'auto', block: 'center'});
        editor.focus({preventScroll: true});
        return true;
    };
    requestAnimationFrame(() => requestAnimationFrame(() => {
        if (focusEditor()) return;
        const observer = new MutationObserver(() => {
            if (focusEditor()) observer.disconnect();
        });
        observer.observe(document.body, {childList: true, subtree: true});
        setTimeout(() => observer.disconnect(), 2000);
    }));
})();
</script>
"""


def focus_chat_composer() -> None:
    st.html(_FOCUS_COMPOSER, unsafe_allow_javascript=True)
