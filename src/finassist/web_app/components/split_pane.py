"""IDE-style divider attached to app-owned pane containers."""

import streamlit as st

# Static application code. No user content or third-party scripts are interpolated.
_SPLIT_PANE = """
<script>
(() => {
    window.finpilotSplitCleanup?.();
    const abort = new AbortController();
    const {signal} = abort;
    let handle, row, left, right, dragging = false;
    let share = Number(sessionStorage.getItem('finpilotChatShare')) || 0.38;
    share = Math.max(0.25, Math.min(0.55, share));
    const narrow = matchMedia('(max-width: 900px)');
    function paint() {
        if (!row || !left?.isConnected) return;
        const stacked = narrow.matches;
        handle.hidden = stacked;
        row.style.height = stacked ? '' : '100%';
        for (const pane of [left, right]) {
            pane.style.height = stacked ? '' : '100%';
            pane.style.minHeight = '0';
        }
        const anchors = [document.querySelector('.st-key-workspace_content'),
            document.querySelector('.st-key-finpilot_chat_panel')];
        anchors.forEach((anchor, index) => {
            let node = anchor;
            while (node && node !== [left, right][index]) {
                node.style.height = stacked ? '' : '100%';
                node.style.minHeight = '0';
                node = node.parentElement;
            }
        });
        if (stacked) {
            left.style.removeProperty('flex');
            right.style.removeProperty('flex');
            return;
        }
        left.style.setProperty('flex', `${1 - share} 1 0%`, 'important');
        right.style.setProperty('flex', `${share} 1 0%`, 'important');
        const edge = left.getBoundingClientRect().right;
        const gap = right.getBoundingClientRect().left - edge;
        handle.style.left = `${edge - row.getBoundingClientRect().left + gap / 2}px`;
        const bounds = row.getBoundingClientRect();
        const visibleTop = Math.max(0, bounds.top);
        const visibleBottom = Math.min(window.innerHeight, bounds.bottom);
        const center = (visibleTop + visibleBottom) / 2 - bounds.top;
        handle.style.setProperty('--grip-top', `${center - 10}px`);
        handle.setAttribute('aria-valuenow', String(Math.round(share * 100)));
    }
    function save() {
        sessionStorage.setItem('finpilotChatShare', String(share));
        paint();
    }
    function install() {
        const content = document.querySelector('.st-key-workspace_content');
        const chat = document.querySelector('.st-key-finpilot_chat_panel');
        if (!content || !chat) return;
        if (handle?.isConnected && left?.contains(content) && right?.contains(chat)) return;
        handle?.remove();
        left = content;
        while (left.parentElement && !left.parentElement.contains(chat)) left = left.parentElement;
        row = left.parentElement;
        if (!row) return;
        right = [...row.children].find(child => child.contains(chat));
        if (!right || right === left) return;
        row.style.position = 'relative';
        handle = document.createElement('div');
        handle.className = 'finpilot-split-handle';
        handle.tabIndex = 0;
        handle.setAttribute('role', 'separator');
        handle.setAttribute('aria-label', 'Resize chat panel');
        handle.setAttribute('aria-orientation', 'vertical');
        handle.setAttribute('aria-valuemin', '25');
        handle.setAttribute('aria-valuemax', '55');
        handle.title = 'Drag to resize chat. Arrow keys resize; double-click resets.';
        row.append(handle);
        handle.addEventListener('pointerdown', event => {
            if (event.button !== 0) return;
            dragging = true;
            handle.focus({preventScroll: true});
            handle.setPointerCapture(event.pointerId);
            event.preventDefault();
        }, {signal});
        handle.addEventListener('pointermove', event => {
            if (!dragging) return;
            const bounds = row.getBoundingClientRect();
            share = Math.max(0.25, Math.min(0.55, (bounds.right - event.clientX) / bounds.width));
            save();
        }, {signal});
        const stop = () => { dragging = false; };
        handle.addEventListener('pointerup', stop, {signal});
        handle.addEventListener('lostpointercapture', stop, {signal});
        handle.addEventListener('dblclick', () => { share = 0.38; save(); }, {signal});
        handle.addEventListener('keydown', event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return;
            event.preventDefault();
            share = event.key === 'Home' ? 0.38 : Math.max(0.25, Math.min(0.55,
                share + (event.key === 'ArrowLeft' ? 0.02 : -0.02)));
            save();
        }, {signal});
        paint();
    }
    const observer = new MutationObserver(() => { install(); paint(); });
    observer.observe(document.body, {childList: true, subtree: true});
    window.addEventListener('resize', paint, {signal});
    window.addEventListener('scroll', paint, {signal, capture: true, passive: true});
    narrow.addEventListener('change', paint, {signal});
    window.finpilotSplitCleanup = () => {
        observer.disconnect();
        abort.abort();
        handle?.remove();
    };
    install();
})();
</script>
<style>
.finpilot-split-handle {
    position: absolute; top: 0; bottom: 0; width: 16px;
    transform: translateX(-50%); cursor: col-resize; z-index: 5;
    touch-action: none; border-radius: 6px;
}
.finpilot-split-handle::before {
    content: ''; position: absolute; top: var(--grip-top, 50%); left: 6px;
    width: 4px; height: 4px; border-radius: 50%;
    background: #475569;
    box-shadow: 0 8px currentColor, 0 16px currentColor;
    color: #475569;
}
.finpilot-split-handle:hover::before,
.finpilot-split-handle:focus-visible::before {
    background: #0F766E; color: #0F766E;
}
.finpilot-split-handle:focus-visible { outline: none; }
.finpilot-split-handle:focus-visible::before {
    background: #153E75; color: #153E75;
    transform: scale(1.3);
}
</style>
"""


def resizable_chat_divider() -> None:
    """Resize native panes in-browser without rerunning or clearing chat state."""
    st.html(_SPLIT_PANE, unsafe_allow_javascript=True)
