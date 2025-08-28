document.addEventListener("DOMContentLoaded", function () {
  console.log("DOM fully loaded");

  particlesJS.load("particles-js", "/static/particles.json", function () {
    console.log("particles.js loaded - callback");
  });

  const statusElement = document.getElementById("status");
  if (statusElement) {
    const status = statusElement.innerText.split(": ")[1];
    const jobId = window.location.pathname.split("/").pop();

    function checkStatus() {
      fetch("/job_status/" + jobId)
        .then((response) => response.json())
        .then((data) => {
          statusElement.innerText = "Status: " + data.status;
          if (data.status === "COMPLETED") {
            document.getElementById("loading-animation").style.display = "none";
            location.reload();
          } else if (data.status !== "FAILED") {
            setTimeout(checkStatus, 2000);
          } else {
            document.getElementById("loading-animation").style.display = "none";
          }
        });
    }

    if (status !== "COMPLETED" && status !== "FAILED") {
      document.getElementById("loading-animation").style.display =
        "inline-block";
      checkStatus();
    } else {
      document.getElementById("loading-animation").style.display = "none";
    }
  }

  // Display selected filename
  const fileInput = document.getElementById("file");
  const fileLabel = document.querySelector(".file-label");
  if (fileInput && fileLabel) {
    fileInput.addEventListener("change", function () {
      if (this.files && this.files.length > 0) {
        fileLabel.textContent = this.files[0].name;
      } else {
        fileLabel.textContent = "Picture";
      }
    });
  }

  // Profile selection functionality
  const selectionForm = document.getElementById("selection-form");
  if (selectionForm) {
    initializeProfileSelection();
  }
});

// Profile Selection Functions
function initializeProfileSelection() {
  const checkboxes = document.querySelectorAll('.profile-checkbox');
  const selectAllBtn = document.getElementById('select-all-btn');
  const selectNoneBtn = document.getElementById('select-none-btn');
  const createGifBtn = document.getElementById('create-gif-btn');
  const selectionCount = document.getElementById('selection-count');
  const submitHelp = document.querySelector('.submit-help');

  // Update selection count and button state
  function updateSelectionState() {
    const selectedCount = document.querySelectorAll('.profile-checkbox:checked').length;
    const totalCount = checkboxes.length;
    
    // Update count display
    if (selectionCount) {
      selectionCount.textContent = `${selectedCount} selected`;
    }
    
    // Update submit button state
    if (createGifBtn) {
      if (selectedCount > 0) {
        createGifBtn.disabled = false;
        createGifBtn.textContent = `Create GIF from ${selectedCount} Selected Profile${selectedCount > 1 ? 's' : ''}`;
        if (submitHelp) {
          submitHelp.textContent = `Ready to create GIF from ${selectedCount} profile${selectedCount > 1 ? 's' : ''}.`;
        }
      } else {
        createGifBtn.disabled = true;
        createGifBtn.textContent = 'Create GIF from Selected Profiles';
        if (submitHelp) {
          submitHelp.textContent = 'Please select at least one profile to continue.';
        }
      }
    }
    
    // Update select all/none button states
    if (selectAllBtn) {
      selectAllBtn.textContent = selectedCount === totalCount ? 'All Selected' : 'Select All';
      selectAllBtn.disabled = selectedCount === totalCount;
    }
    
    if (selectNoneBtn) {
      selectNoneBtn.textContent = selectedCount === 0 ? 'None Selected' : 'Select None';
      selectNoneBtn.disabled = selectedCount === 0;
    }
  }

  // Add event listeners to checkboxes
  checkboxes.forEach(checkbox => {
    checkbox.addEventListener('change', updateSelectionState);
  });

  // Select all button functionality
  if (selectAllBtn) {
    selectAllBtn.addEventListener('click', function() {
      checkboxes.forEach(checkbox => {
        checkbox.checked = true;
      });
      updateSelectionState();
    });
  }

  // Select none button functionality
  if (selectNoneBtn) {
    selectNoneBtn.addEventListener('click', function() {
      checkboxes.forEach(checkbox => {
        checkbox.checked = false;
      });
      updateSelectionState();
    });
  }

  // Form submission confirmation
  const selectionForm = document.getElementById('selection-form');
  if (selectionForm) {
    selectionForm.addEventListener('submit', function(e) {
      const selectedCount = document.querySelectorAll('.profile-checkbox:checked').length;
      
      if (selectedCount === 0) {
        e.preventDefault();
        alert('Please select at least one profile to create a GIF.');
        return false;
      }
      
      // Show confirmation dialog
      const confirmMessage = `You've selected ${selectedCount} profile${selectedCount > 1 ? 's' : ''}. ` +
                            `This will create ${Math.ceil(selectedCount / 100)} GIF file${Math.ceil(selectedCount / 100) > 1 ? 's' : ''}. ` +
                            `Do you want to continue?`;
      
      if (!confirm(confirmMessage)) {
        e.preventDefault();
        return false;
      }
      
      // Show processing state
      if (createGifBtn) {
        createGifBtn.disabled = true;
        createGifBtn.textContent = 'Creating GIF... Please wait';
      }
    });
  }

  // Initialize the selection state
  updateSelectionState();

  // Add keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Ctrl+A or Cmd+A to select all
    if ((e.ctrlKey || e.metaKey) && e.key === 'a' && !e.target.matches('input[type="text"], textarea')) {
      e.preventDefault();
      if (selectAllBtn && !selectAllBtn.disabled) {
        selectAllBtn.click();
      }
    }
    
    // Escape to deselect all
    if (e.key === 'Escape') {
      if (selectNoneBtn && !selectNoneBtn.disabled) {
        selectNoneBtn.click();
      }
    }
  });
}
