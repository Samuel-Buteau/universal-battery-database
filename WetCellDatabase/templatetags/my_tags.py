from django import template

register = template.Library()


@register.filter
def index(indexable, i):
    if i >= len(indexable):
        return None
    else:
        return indexable[i]

@register.simple_tag(name='get_obj_attr')
def get_obj_attr(obj, attr, incept):
    return attr.format(incept)


@register.filter(name='times')
def times(number):
    return range(number)