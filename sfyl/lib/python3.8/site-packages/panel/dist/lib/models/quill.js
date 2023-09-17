var _a;
import { div } from "@bokehjs/core/dom";
import { HTMLBox, HTMLBoxView } from "./layout";
class QuillInputView extends HTMLBoxView {
    connect_signals() {
        super.connect_signals();
        this.connect(this.model.properties.disabled.change, () => this.quill.enable(!this.model.disabled));
        this.connect(this.model.properties.text.change, () => {
            if (this._editing)
                return;
            this._editing = true;
            this.quill.enable(false);
            this.quill.setContents([]);
            this.quill.clipboard.dangerouslyPasteHTML(this.model.text);
            this.quill.enable(!this.model.disabled);
            this._editing = false;
        });
        const { mode, toolbar, placeholder } = this.model.properties;
        this.on_change([placeholder], () => {
            this.quill.root.setAttribute('data-placeholder', this.model.placeholder);
        });
        this.on_change([mode, toolbar], () => {
            this.render();
            this._layout_toolbar();
        });
    }
    _layout_toolbar() {
        if (this._toolbar == null) {
            this.el.style.removeProperty('padding-top');
        }
        else {
            const height = this._toolbar.getBoundingClientRect().height + 1;
            this.el.style.paddingTop = height + "px";
            this._toolbar.style.marginTop = -height + "px";
        }
    }
    render() {
        super.render();
        this.container = div({ style: "visibility: hidden;" });
        this.shadow_el.appendChild(this.container);
        const theme = (this.model.mode === 'bubble') ? 'bubble' : 'snow';
        this.watch_stylesheets();
        this.quill = new window.Quill(this.container, {
            modules: {
                toolbar: this.model.toolbar
            },
            readOnly: true,
            placeholder: this.model.placeholder,
            theme: theme
        });
        this._editor = this.shadow_el.querySelector('.ql-editor');
        this._toolbar = this.shadow_el.querySelector('.ql-toolbar');
        this.quill.clipboard.dangerouslyPasteHTML(this.model.text);
        this.quill.on('text-change', () => {
            if (this._editing)
                return;
            this._editing = true;
            this.model.text = this._editor.innerHTML;
            this._editing = false;
        });
        if (!this.model.disabled)
            this.quill.enable(!this.model.disabled);
        document.addEventListener("selectionchange", (..._args) => {
            // Update selection and some other properties
            this.quill.selection.update();
        });
    }
    style_redraw() {
        this.container.style.visibility = 'visible';
        this.invalidate_layout();
    }
    after_layout() {
        super.after_layout();
        this._layout_toolbar();
    }
}
QuillInputView.__name__ = "QuillInputView";
export { QuillInputView };
class QuillInput extends HTMLBox {
    constructor(attrs) {
        super(attrs);
    }
}
_a = QuillInput;
QuillInput.__name__ = "QuillInput";
QuillInput.__module__ = "panel.models.quill";
(() => {
    _a.prototype.default_view = QuillInputView;
    _a.define(({ Any, String }) => ({
        mode: [String, 'toolbar'],
        placeholder: [String, ''],
        text: [String, ''],
        toolbar: [Any, null],
    }));
    _a.override({
        height: 300
    });
})();
export { QuillInput };
//# sourceMappingURL=quill.js.map