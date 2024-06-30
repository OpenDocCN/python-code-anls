# `D:\src\scipysrc\scikit-learn\doc\js\scripts\dropdown.js`

```
/**
 * This script is used to enhance sphinx-design dropdowns by dynamically adding buttons
 * for collapsing/expanding all dropdowns on the page. This is necessary because some
 * browsers (e.g., Firefox) cannot search into collapsed <details> elements.
 *
 * Buttons are added dynamically with JavaScript to ensure they only appear and function
 * when JavaScript is enabled, avoiding unnecessary display when it's not.
 */

function addToggleAllButtons() {
  // Get all sphinx-design dropdowns on the page
  const allDropdowns = document.querySelectorAll("details.sd-dropdown");

  function collapseAll() {
    // Function to collapse all dropdowns on the page
    console.log("[SK] Collapsing all dropdowns...");
    allDropdowns.forEach((dropdown) => {
      dropdown.removeAttribute("open");
    });
  }

  function expandAll() {
    // Function to expand all dropdowns on the page
    console.log("[SK] Expanding all dropdowns...");
    allDropdowns.forEach((dropdown) => {
      dropdown.setAttribute("open", "");
    });
  }

  // Configuration for the collapse and expand buttons
  const buttonConfigs = new Map([
    ["up", { desc: "Collapse", action: collapseAll }],
    ["down", { desc: "Expand", action: expandAll }],
  ]);

  // Iterate through each sphinx-design dropdown
  allDropdowns.forEach((dropdown) => {
    // Get the summary element of the dropdown where buttons will be added
    const summaryTitle = dropdown.querySelector("summary.sd-summary-title");

    // Iterate through button configurations (up and down)
    for (const [direction, config] of buttonConfigs) {
      // Create a new button element with an icon
      var newButton = document.createElement("button");
      var newIcon = document.createElement("i");
      newIcon.classList.add("fa-solid", `fa-angles-${direction}`);
      newButton.appendChild(newIcon);
      
      // Add classes for styling: `sd-summary-up/down` for sphinx-design styling,
      // `sk-toggle-all` for additional custom styling
      newButton.classList.add(`sd-summary-${direction}`, `sk-toggle-all`);
      
      // Configure Bootstrap tooltip settings
      newButton.setAttribute("data-bs-toggle", "tooltip");
      newButton.setAttribute("data-bs-placement", "top");
      newButton.setAttribute("data-bs-offset", "0,10");
      newButton.setAttribute("data-bs-title", `${config.desc} all dropdowns`);
      
      // Assign the collapse/expand action to the button
      newButton.onclick = config.action;
      
      // Append the newly created button to the summary element of the dropdown
      summaryTitle.appendChild(newButton);
    }
  });
}

// Execute the function when the DOM content is fully loaded
document.addEventListener("DOMContentLoaded", addToggleAllButtons);
```