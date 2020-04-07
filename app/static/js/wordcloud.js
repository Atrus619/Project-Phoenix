$( document ).ready(function() {
  reHeightWordcloud();
});

$( window ).resize(function() {
  reHeightWordcloud();
});

function reHeightWordcloud() {
  var captionHeight = $('#caption').outerHeight();
  var windowHeight = $(window).outerHeight();
  // Magic 4 prevents vertical scroll bar from appearing
  $('#wordcloud').height(windowHeight - captionHeight - 4);
};
