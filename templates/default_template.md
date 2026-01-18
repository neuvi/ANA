---
title: "{{ title }}"
tags: {{ tags }}
type: {{ category }}
created: {{ created }}
source: "ANA-generated"
{% for key, value in extra_metadata.items() %}
{{ key }}: {{ value }}
{% endfor %}
---

# {{ title }}

{{ content }}

---

## Related Links
{% for link in suggested_links %}
- [[{{ link }}]]
{% endfor %}
