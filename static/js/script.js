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
});
