const calculateCalories = () => {
  const gender = document.querySelector('input[name="gender"]:checked').value;
  const exercise = document.querySelector('select[name="exercise"]').value;
  const duration = document.querySelector('input[name="duration"]').value;
  const heartRate = document.querySelector(
    'input[name="heart_rate_range"]'
  ).value;

  const url = `/predict?gender=${gender}&exercise=${exercise}&duration=${duration}&heart_rate_range=${heartRate}`;

  fetch(url)
    .then((response) => response.json())
    .then((data) => {
      const predictionEl = document.querySelector(".prediction");
      predictionEl.innerHTML = `<p>You will burn around <span class="calories">${data.prediction}</span> Calories.</p>`;
    })
    .catch((error) => console.error(error));
};

// Call calculateCalories when the page loads
document.addEventListener("DOMContentLoaded", () => {
  calculateCalories();
});

// Call calculateCalories when the user changes any of the input values
document.querySelectorAll("input, select").forEach((input) => {
  input.addEventListener("input", calculateCalories);
});
