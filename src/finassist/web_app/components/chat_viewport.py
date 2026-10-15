"""Keep the native transcript scroller above an always-visible composer."""

import streamlit as st

_VIEWPORT = """
<script>
(() => {
    window.finpilotChatViewportCleanup?.();
    const abort = new AbortController();
    let frame, scrollFrame;
    let scrollPending = __SCROLL_LATEST__;
    function resize() {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
            const history = document.querySelector('.st-key-chat_history');
            const composer = document.querySelector('.st-key-chat_composer');
            if (!history || !composer) return;
            const frame = document.querySelector('.st-key-workspace_body');
            const desktop = matchMedia('(min-width: 901px)').matches;
            if (frame) {
                const remaining = window.innerHeight - frame.getBoundingClientRect().top - 16;
                const height = desktop ? `${Math.max(220, remaining)}px` : '';
                if (frame.style.height !== height) frame.style.height = height;
                frame.style.maxHeight = height;
                frame.style.minHeight = '0';
                frame.style.setProperty('--pane-height', `${Math.max(186, remaining - 34)}px`);
                frame.style.flexShrink = '0';
                const content = document.querySelector('.st-key-workspace_content');
                let scrollArea = content;
                while (scrollArea && scrollArea !== frame &&
                    !/(auto|scroll)/.test(getComputedStyle(scrollArea).overflowY)) {
                    scrollArea = scrollArea.parentElement;
                }
                if (scrollArea && scrollArea !== frame) {
                    scrollArea.style.height = desktop ?
                        `${Math.max(186, remaining - 34)}px` : '500px';
                    scrollArea.style.maxHeight = scrollArea.style.height;
                }
            }
            const scope = document.querySelector('.st-key-chat_scope');
            const footer = composer.getBoundingClientRect().height +
                (scope?.getBoundingClientRect().height || 0) + 48;
            const panel = document.querySelector('.st-key-finpilot_chat_panel');
            const bottom = desktop && panel ?
                Math.min(window.innerHeight, panel.getBoundingClientRect().bottom) :
                window.innerHeight;
            const available = bottom - history.getBoundingClientRect().top - footer;
            const height = `${Math.max(100, Math.floor(available))}px`;
            if (history.style.height !== height) history.style.height = height;
            const latest = document.querySelector('.st-key-latest_chat_turn___TURN_REVISION__');
            if (latest && latest.children.length) {
                // Reserve enough space below a short turn to align its question at the top.
                latest.style.minHeight = `${Math.max(0, history.clientHeight - 8)}px`;
                if (scrollPending) {
                    cancelAnimationFrame(scrollFrame);
                    scrollFrame = requestAnimationFrame(() => {
                        scrollFrame = requestAnimationFrame(() => {
                            if (!latest.isConnected || !history.isConnected) return;
                            // Native Streamlit containers can wrap their scrolling element.
                            let scroller = latest.parentElement;
                            while (scroller && scroller !== history.parentElement &&
                                !/(auto|scroll)/.test(getComputedStyle(scroller).overflowY)) {
                                scroller = scroller.parentElement;
                            }
                            if (!scroller || scroller === history.parentElement) scroller = history;
                            scroller.scrollTop += latest.getBoundingClientRect().top -
                                scroller.getBoundingClientRect().top - scroller.clientTop;
                            scrollPending = false;
                        });
                    });
                }
            }
        });
    }
    // Streamlit commits new message containers asynchronously after the HTML delta.
    const mutations = new MutationObserver(resize);
    mutations.observe(document.body, {childList: true, subtree: true});
    const observer = new ResizeObserver(resize);
    const panel = document.querySelector('.st-key-finpilot_chat_panel');
    if (panel) observer.observe(panel);
    window.addEventListener('resize', resize, {signal: abort.signal});
    window.addEventListener('scroll', resize, {signal: abort.signal, capture: true, passive: true});
    window.finpilotChatViewportCleanup = () => {
        abort.abort();
        observer.disconnect();
        mutations.disconnect();
        cancelAnimationFrame(frame);
        cancelAnimationFrame(scrollFrame);
    };
    resize();
})();
</script>
"""


def fit_chat_viewport(*, scroll_to_latest: bool = False, turn_revision: int = 0) -> None:
    st.html(
        _VIEWPORT.replace("__SCROLL_LATEST__", "true" if scroll_to_latest else "false").replace(
            "__TURN_REVISION__", str(turn_revision)
        ),
        unsafe_allow_javascript=True,
    )
