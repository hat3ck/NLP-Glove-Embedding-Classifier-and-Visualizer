// easy d3-based word cloud plugin https://github.com/wvengen/d3-wordcloud
// requires https://github.com/jasondavies/d3-cloud
// based on https://github.com/shprink/d3js-wordcloud

function create_vis(words){
  d3.wordcloud()
    .size([1000, 550])
    // .fill(d3.scale.ordinal().range(["#884400", "#448800", "#888800", "#444400"]))
    .words(words)
    .onwordclick(function(d, i) {
      if (d.href) { window.location = d.href; }
    })
    .start();
  }
(function() {
  function wordcloud() {
    var selector = '#wordcloud',
        element = d3.select(selector),
        transitionDuration = 200,
        scale = 'sqrt',
        fill = d3.scale.category20b(),
        layout = d3.layout.cloud(),
        fontSize = null,
        svg = null,
        vis = null,
        margin = 150,
        onwordclick = undefined;

    wordcloud.element = function(x) {
      if (!arguments.length) return element;
      element = x == null ? '#wordcloud' : x;
      return wordcloud
    };

    wordcloud.selector = function(x) {
      if (!arguments.length) return selector;
      element = d3.select(x == null ? selector : x);
      return wordcloud;
    };

    wordcloud.transitionDuration = function(x) {
      if (!arguments.length) return transitionDuration;
      transitionDuration = typeof x == 'function' ? x() : x;
      return wordcloud;
    };

    wordcloud.scale = function(x) {
      if (!arguments.length) return scale;
      scale = x == null ? 'sqrt' : x;
      return wordcloud;
    };

    wordcloud.fill = function(x) {
      if (!arguments.length) return fill;
      fill = x == null ? d3.scale.category20b() : x;
      return wordcloud;
    };

    wordcloud.onwordclick = function (func) {
        onwordclick = func;
        return wordcloud;
    }

    wordcloud.start = function() {
      init();
      layout.start(arguments);
      return wordcloud;
    };

    function init() {
      layout
        .fontSize(function(d) {
          return fontSize(+d.size);
        })
        .text(function(d) {
          return d.text;
        })
        .on("end", draw);

      svg = element.append("svg");
      vis = svg.append("g").attr("transform", "translate(" + [layout.size()[0] >> 1, layout.size()[1] >> 1] + ")");

      update();
      svg.on('resize', function() { update() });
    }

    function draw(data, bounds) {
      var w = layout.size()[0]-margin,
          h = layout.size()[1]-margin;

      svg.attr("width", w).attr("height", h);

      scaling = bounds ? Math.min(
        w / Math.abs(bounds[1].x - w / 2),
        w / Math.abs(bounds[0].x - w / 2),
        h / Math.abs(bounds[1].y - h / 2),
        h / Math.abs(bounds[0].y - h / 2)) / 2 : 1;

      var text = vis.selectAll("text")
        .data(data, function(d) {
          return d.text.toLowerCase();
        });
      text.transition()
        .duration(transitionDuration)
        .attr("transform", function(d) {
          return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
        })
        .style("font-size", function(d) {
          return d.size + "px";
        });
      text.enter().append("text")
        .attr("text-anchor", "middle")
        .attr("transform", function(d) {
          return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
        })
        .style("font-size", function(d) {
          return d.size + "px";
        })
        .style("opacity", 1e-6)
        .transition()
        .duration(transitionDuration)
        .style("opacity", 1);
      text.style("font-family", function(d) {
          return d.font || layout.font() || svg.style("font-family");
        })
        .style("fill", function(d){
          return colorvar(d.text);
        }
        // function(d) {
        //   return fill(d.text.toLowerCase());
        // }
      )
        .text(function(d) {
          return d.text;
        })
        // clickable words
        .style("cursor", function(d, i) {
          if (onwordclick !== undefined) return 'pointer';
        })
        .on("mouseover", function(d, i) {
          if (onwordclick !== undefined) {
            d3.select(this).transition().style('font-size', d.size + 10 + 'px');
            handleMouseOver(d);
          }
        })
        .on("mouseout", function(d, i) {
          if (onwordclick !== undefined) {
            d3.select(this).transition().style('font-size', d.size + 'px');
            handleMouseOut(d);
          }
        })
        .on("click", function(d, i) {
          if (onwordclick !== undefined) {
                onwordclick(d,i);
                // Used code from here: http://fabien-d.github.io/alertify.js/
                alertify.alert(alert_text(d.text));
                // alert(alert_text(d.tex));
            }
        });
        function handleMouseOver(d) {
          var group = vis.append('g')
                           .attr('id', 'story-titles');
           var base = d.y - d.size ;
           var x = d.x;
           var second_word_margin = 15;
           var y = base - (14+second_word_margin);
           console.log(base , d.y,d.size);
          group.selectAll('text')
               .data('customdata')
               .enter().append('text')
               // .attr('x', d.x)
               // .attr('y', base - d.y/14 )
               .attr('text-anchor', 'middle')
               .attr('font-size','1em')
               .append("tspan")
               .attr('x',x)
               .attr('y', y)
               .text('occourrence:'+ tooltip_text(d.text)[0])
               .append("tspan")
               .attr('x',x)
               .attr('y',y+second_word_margin)
               .text('fanboy: '+tooltip_text(d.text)[1]+'%');
               // .attr('fill','white');

          var bbox = group.node().getBBox();
          var bboxPadding = 5;

          // place a white background to see text more clearly
          var rect = group.insert('rect', ':first-child')
                        .attr('x', bbox.x)
                        .attr('y', bbox.y)
                        .attr('width', bbox.width + bboxPadding)
                        .attr('height', bbox.height + bboxPadding)
                        .attr('rx', 10)
                        .attr('ry', 10)
                        .attr('class', 'label-background-strong');
        }

        function handleMouseOut(d) {
          d3.select('#story-titles').remove();
        }
      vis.transition()
        .attr("transform", "translate(" + [w >> 1, h >> 1] + ")scale(" + scaling + ")");
    };


    function update() {
      var words = layout.words();
      fontSize = d3.scale[scale]().range([10, 100]);
      if (words.length) {
        fontSize.domain([+words[words.length - 1].size || 1, +words[0].size]);
      }
    }

    return d3.rebind(wordcloud, layout, 'on', 'words', 'size', 'font', 'fontStyle', 'fontWeight', 'spiral', 'padding');
  }

  if (typeof module === "object" && module.exports) module.exports = wordcloud;
  else d3.wordcloud = wordcloud;
})();
