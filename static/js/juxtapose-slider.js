function createSlider(elementId, images, options) {
  console.log(`Creating slider for ${elementId} with images:`, images); // Debugging line
  return new juxtapose.JXSlider(elementId, images, options);
}

sliderOptions = {
  animate: true,
  showLabels: true,
  showCredits: false,
  startingPosition: "35%",
  makeResponsive: true
};

// Function to initialize sliders with image data
function initializeSliders(sliders) {
  sliders.forEach(slider => {
    createSlider(slider.elementId, slider.images, slider.options);
  });
}