(function () {
  // Generic row toggle — used by source_pipeline and output_quality
  window.toggle = function (id) {
    var el = document.getElementById(id);
    if (el) el.style.display = el.style.display === 'none' ? 'table-row' : 'none';
  };
})();
