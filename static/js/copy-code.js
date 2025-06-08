document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('pre > code').forEach((codeBlock) => {
    const pre = codeBlock.parentNode;

    // Create wrapper div
    const wrapper = document.createElement('div');
    wrapper.className = 'code-wrapper';

    // Create header bar
    const header = document.createElement('div');
    header.className = 'code-header';

    // Language label (e.g. cpp, python)
    const langMatch = codeBlock.className.match(/language-(\w+)/);
    const lang = langMatch ? langMatch[1] : '';
    const label = document.createElement('span');
    label.className = 'language-label';
    label.textContent = lang;

    // Copy button
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.innerText = 'Copy';

    button.addEventListener('click', () => {
      const code = codeBlock.innerText;
      navigator.clipboard.writeText(code).then(() => {
        button.innerText = 'Copied!';
        setTimeout(() => { button.innerText = 'Copy'; }, 2000);
      });
    });

    // Assemble header
    header.appendChild(label);
    header.appendChild(button);

    // Move pre into wrapper
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(header);
    wrapper.appendChild(pre);
  });
});
