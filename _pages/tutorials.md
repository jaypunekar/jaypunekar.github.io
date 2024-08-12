---
layout: archive
title: "Tutorials"
permalink: /tutorials/
toc: true
author_profile: true
---
{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}
On this tutorials page, I have meticulously crafted step-by-step guides on building a drone show, designing a budget-friendly, industry-grade planetarium, and VR synchronization. These are highly specialized topics with limited information available online, particularly when it comes to drone shows and planetarium setups. My aim is to provide clear, comprehensive instructions that fill the gap in these niche areas, making them accessible to enthusiasts and professionals alike.
{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

