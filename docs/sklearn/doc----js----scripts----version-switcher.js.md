# `D:\src\scipysrc\scikit-learn\doc\js\scripts\version-switcher.js`

```
/**
 * Adds the link to available documentation page as the last entry in the version
 * switcher dropdown. Since other entries in the dropdown are also added dynamically,
 * we only add the link when the user clicks on some version switcher button to make
 * sure that this entry is the last one.
 */
function addVersionSwitcherAvailDocsLink() {
    // Flag to track whether the availability docs link has been added
    var availDocsLinkAdded = false;

    // Iterate through all version switcher buttons
    document
        .querySelectorAll(".version-switcher__button")
        .forEach(function (btn) {
            // Add click event listener to each button
            btn.addEventListener("click", function () {
                // Check if the availability docs link has not been added yet
                if (!availDocsLinkAdded) {
                    // Update all version switcher dropdown menus when any button is clicked
                    document
                        .querySelectorAll(".version-switcher__menu")
                        .forEach(function (menu) {
                            // Create a new link element for availability docs
                            var availDocsLink = document.createElement("a");
                            availDocsLink.setAttribute(
                                "href",
                                "https://scikit-learn.org/dev/versions.html"
                            );
                            availDocsLink.innerHTML = "More";
                            // Use the same class as the last existing entry for consistency
                            availDocsLink.className = menu.lastChild.className;
                            // Add a specific class for identification
                            availDocsLink.classList.add("sk-avail-docs-link");
                            // Append the availability docs link as the last child of the menu
                            menu.appendChild(availDocsLink);
                        });
                    // Set flag to true to prevent adding the link again
                    availDocsLinkAdded = true;
                }
            });
        });
}

// Event listener to add availability docs link when DOM content is loaded
document.addEventListener("DOMContentLoaded", addVersionSwitcherAvailDocsLink);
```