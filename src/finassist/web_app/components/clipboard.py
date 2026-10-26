"""Browser-local clipboard control; message text is escaped as data, never code."""

from html import escape

import streamlit as st


def copy_message(text: str, *, display_text: bool = False) -> None:
    content = (
        '<span style="white-space:pre-wrap;overflow-wrap:anywhere;flex:1;min-width:0">'
        + escape(text) + '</span>' if display_text else ""
    )
    st.html(
        '<div style="display:flex;align-items:flex-start;gap:8px">' + content
        + '<button type="button" class="finpilot-copy" aria-label="Copy message" '
        'title="Copy message" data-copy="' + escape(text, quote=True) + '">'
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.7" aria-hidden="true">'
        '<rect x="8" y="8" width="12" height="12" rx="2"/>'
        '<path d="M16 8V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h4"/>'
        '</svg></button></div>'
        '<style>.finpilot-copy{background:transparent;border:0;color:inherit;cursor:pointer;'
        'padding:6px;border-radius:5px}.finpilot-copy:hover{background:#e8eef5}</style>'
        """
        <script>
        document.querySelectorAll('.finpilot-copy:not([data-bound])').forEach(button => {
            button.dataset.bound = 'true';
            button.addEventListener('click', async () => {
                try {
                    if (navigator.clipboard && window.isSecureContext) {
                        await navigator.clipboard.writeText(button.dataset.copy);
                    } else {
                        const field = document.createElement('textarea');
                        field.value = button.dataset.copy;
                        field.style.position = 'fixed';
                        field.style.opacity = '0';
                        document.body.appendChild(field);
                        try {
                            field.select();
                            if (!document.execCommand('copy')) throw new Error('Copy failed');
                        } finally { field.remove(); }
                        button.focus();
                    }
                    button.title = 'Copied!';
                    button.setAttribute('aria-label', 'Copied!');
                    setTimeout(() => {
                        button.title = 'Copy message';
                        button.setAttribute('aria-label', 'Copy message');
                    }, 2000);
                } catch {
                    button.title = 'Copy unavailable. Select and copy the message text.';
                }
            });
        });
        </script>
        """,
        unsafe_allow_javascript=True,
    )
