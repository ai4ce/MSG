function copyToClipboard() {
    const bibtexElement = document.getElementById("bibtexInfo");

    // Create a range object to select the Bibtex info
    const range = document.createRange();
    range.selectNode(bibtexElement);

    // Select the range
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);

    // Copy the selected text to the clipboard
    document.execCommand("copy");

    // Clean up the selection
    selection.removeAllRanges();
}
