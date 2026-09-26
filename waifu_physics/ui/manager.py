"""The chain manager: a floating, file-browser-style tree of the active armature's groups (folders) and
chains, drawn over the 3D viewport and opened from the sidebar.

Click a chain to select it (its bones are selected in the viewport), Shift-click a range, Ctrl-click to add
or drop one. Drag chains onto a group to move them there, or onto empty space for a new group; drag a group
onto another to merge them. It edits one armature, the active one: a group never spans armatures.

Opt-in detail, closed until opened: a chain's arrow lists its bones, which select, drag and delete like
chains (a run of bones moves as a group of its own; deleting a bone removes it and all below). Each group's
Links row lists its links: clicking one picks it (red in the viewport), its x removes it.

Input comes through a 2D gizmo covering the manager, so clicks on it never reach the viewport, and nothing
outside it is touched. No modal operator runs while it is open (a running one would hold off autosave)."""
import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

from ..data import links as chain_links
from . import ops

ROW = 20                   # a row's height, and the unit of the layout, at a UI scale of 1
WIDTH = 270                # the default width; drag the right edge to change it
MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT = 200, 700, 120
EDGE = 5                   # the grab margin of the resizable edges
BAR = 10                   # the scroll bar's width
DRAG_START = 5             # pixels the mouse moves before a press becomes a drag
NAV_GUTTER = 16            # the strip on the left holding the armature pane's arrow, open or closed
NAV_MIN, NAV_MAX = 60, 260 # an open pane is as wide as its longest name, within these

_open = set()              # areas (as_pointer) showing the manager
_places = {}               # area -> (left, top) in region pixels, once dragged by its title
_scroll = {}               # area -> the first row shown
_hover = {}                # area -> (x, y), the mouse over the manager
_drags = {}                # area -> Drag, while one is on
_sizes = {}                # area -> (width, height) at a UI scale of 1, once resized; height None fits the rows
_pressed = {}              # area -> the kind of item held down, for its pressed look
_boxes = {}                # area -> (x0, y0, x1, y1), the box select being dragged
_shown = {}                # area -> session_uid of the armature the manager shows
_nav_closed = set()        # areas whose armature pane is collapsed to its arrow
_pan = {}                  # area -> trackpad scrolling not yet a whole row
_last_group_press = {}     # area -> (group index, time): a second press on it soon after is a double-click
_editing = {}              # area -> Editing: a group's name being typed in its row
_open_chains = set()       # (armature session_uid, chain root) of chains listing their bones
_addon_keymaps = []        # (keymap, item): the viewport's own double-click and F2 over a group's row


def is_open(area):
    return area is not None and area.as_pointer() in _open


def toggle(area):
    key = area.as_pointer()
    if key in _open:
        _open.discard(key)
    else:
        _open.add(key)
    area.tag_redraw()


def shown_name(root):
    """A chain as the manager lists it: its root bone, without VRoid's J_Sec_ prefix."""
    return root[6:] if root.startswith("J_Sec_") else root


# --------------------------------------------------------------------------- layout (no drawing: testable)

class Item:
    """A row or a control. root: a chain's root, or a bone row's bone; chain: the chain a bone row is listed
    under; index: a link row's link; state: a chain row's selection, "all", "some" or ""."""
    __slots__ = ("kind", "x0", "y0", "x1", "y1", "group", "root", "text", "count", "enabled", "chain", "index",
                 "state")

    def __init__(self, kind, x0, y0, x1, y1, group=-1, root="", text="", count="", enabled=True, chain="",
                 index=-1, state=""):
        self.kind, self.x0, self.y0, self.x1, self.y1 = kind, x0, y0, x1, y1
        self.group, self.root, self.text, self.count, self.enabled = group, root, text, count, enabled
        self.chain, self.index, self.state = chain, index, state

    def contains(self, x, y):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def __repr__(self):
        return f"Item({self.kind}, group={self.group}, root={self.root!r})"


class Layout:
    """Where everything is, top to bottom. Hit-testing and drawing both read this, so they always agree."""

    def __init__(self, obj, region_size, scale, place=None, scroll=0, drag=None, size=None, armatures=(),
                 nav_open=True, measure=None, open_chains=frozenset()):
        """armatures: the ones the pane on the left lists (with none, there is no pane); nav_open: the pane shows
        them, else only its arrow; measure: a text's width in pixels (the drawing's font), else estimated;
        open_chains: the roots of the chains listing their bones."""
        self.obj, self.scale = obj, scale
        self.open_chains = open_chains
        unit = ROW * scale
        width_unscaled, height_unscaled = size if size is not None else (WIDTH, None)
        self.main_width = min(max(width_unscaled, MIN_WIDTH), MAX_WIDTH)
        names = pane_names([armature.name for armature in armatures])
        gutter = NAV_GUTTER * scale if armatures else 0.0
        pane = 0.0
        if armatures and nav_open:
            measure = measure or (lambda text: len(text) * 7.0 * scale)
            longest = max(measure(name) for name in names.values())
            pane = min(max(longest + 20 * scale, NAV_MIN * scale), NAV_MAX * scale)
        nav_width = gutter + pane
        self.nav_open = pane > 0
        width = self.main_width * scale + nav_width
        region_w, region_h = region_size
        left, top = place if place is not None else (10 * scale, region_h - 110 * scale)
        left = min(max(left, 0.0), max(region_w - width, 0.0))
        top = min(max(top, 3 * unit), region_h)
        self.items, self.rows = [], []
        self.unit = unit
        self.chosen = ops.selected_chain_keys(obj) if obj is not None else set()
        self.selected_bones = {pb.name for pb in obj.pose.bones if pb.select} if obj is not None else set()
        chosen_groups = {group for group, _root in self.chosen}
        self.chosen_groups = chosen_groups
        x0, x1 = left, left + width
        pad, edge = 4 * scale, EDGE * scale
        frame_x0, x0 = x0, x0 + nav_width                     # the groups' side starts after the pane
        self.nav_x0, self.nav_x1 = frame_x0, x0

        title = Item("title", frame_x0, top - unit * 1.2, x1, top,
                     text=f"Chains  ·  {obj.name}" if obj is not None else "Chains")
        close = Item("close", x1 - unit, title.y0, x1, title.y1)
        self.title = title
        y = title.y0 - pad
        tools = y - unit
        buttons = []
        if obj is not None:
            third = (self.main_width * scale - 2 * pad - 2 * pad) / 3          # the groups' side, not the pane
            new = Item("new", x0 + pad, tools, x0 + pad + third * 1.25, y, text="New Group",
                       enabled=bool(self.chosen))
            merge = Item("merge", new.x1 + pad, tools, new.x1 + pad + third * 1.1, y, text="Merge",
                         enabled=len(chosen_groups) > 1)
            delete = Item("delete", merge.x1 + pad, tools, x1 - pad, y, text="Delete", enabled=bool(self.chosen))
            buttons = [new, merge, delete]
            y = tools - pad

        # The rows: every group; the chains of the open ones, the bones of open chains; each group's links.
        rows = []
        if obj is not None:
            for index, group in enumerate(obj.waifu_physics.groups):
                rows.append(("group", index, "", group, ""))
                if not group.show_chains:
                    continue
                for root in group.roots:
                    rows.append(("chain", index, root.name, group, ""))
                    if root.name in open_chains:
                        own = [bone.name for bone in group.excluded]
                        rows += [("bone", index, name, group, root.name)
                                 for name in chain_links.chain_subtree(obj, root.name, own)]
                if len(group.links):
                    rows.append(("links", index, "", group, ""))
                    if group.list_links:
                        rows += [("link", index, k, group, "") for k in range(len(group.links))]
        self.dragging_chains = drag is not None and drag.kind in ("chains", "bones")
        footer = unit if self.dragging_chains else 0.0
        note = unit if obj is None else unit * 2.5 if not len(obj.waifu_physics.groups) else 0.0
        spare = unit * (0.6 if self.dragging_chains else 1.0)      # empty space under the rows, to drop on
        body_top = y
        if height_unscaled is not None:                            # resized: the height is the user's
            bottom = max(top - max(height_unscaled, MIN_HEIGHT) * scale, 0.0)
            room = max(int((body_top - note - bottom - footer - 0.5 * unit) // unit), 1)
        else:                                                      # fits the rows, down to the viewport's foot
            room = max(int((body_top - note - footer - spare - pad) // unit), 1 if self.dragging_chains else 3)
        self.total, self.capacity = len(rows), room
        self.first = min(max(scroll, 0), max(len(rows) - room, 0))
        shown = rows[self.first:self.first + room]
        self.overflow = len(rows) > room
        row_right = x1 - edge - BAR * scale - 2 * scale if self.overflow else x1
        for kind, index, root, group, chain in shown:
            y0 = y - unit
            if kind == "group":
                row = Item("group", x0, y0, row_right, y, group=index, text=group.name,
                           count=str(len(group.roots)))
                self.items.append(Item("fold", x0, y0, x0 + pad + unit, y, group=index))
            elif kind == "chain":
                bones = chain_links.chain_subtree(obj, root, [bone.name for bone in group.excluded])
                picked = sum(name in self.selected_bones for name in bones)
                row = Item("chain", x0, y0, row_right, y, group=index, root=root, text=shown_name(root),
                           count=str(len(bones)),
                           state="all" if bones and picked == len(bones) else "some" if picked else "")
                self.items.append(Item("chain_fold", x0 + 20 * scale, y0, x0 + 36 * scale, y, group=index,
                                       root=root))
            elif kind == "bone":
                linked = sum((link.bone_a == root) + (link.bone_b == root) for link in group.links)
                row = Item("bone", x0, y0, row_right, y, group=index, root=root, text=shown_name(root),
                           count=f"\u2194 {linked}" if linked else "", chain=chain)
            elif kind == "links":
                row = Item("links", x0, y0, row_right, y, group=index, text="Links", count=str(len(group.links)))
            else:
                link = group.links[root]
                row = Item("link", x0, y0, row_right, y, group=index, index=root,
                           text=f"{shown_name(link.bone_a)}  \u2194  {shown_name(link.bone_b)}")
                self.items.append(Item("link_remove", row_right - unit, y0, row_right, y, group=index, index=root))
            self.rows.append(row)
            y = y0
        scroll_bottom = y
        if self.dragging_chains:
            zone = Item("newzone", x0 + pad, y - unit, row_right - pad, y,
                        text="Drop here for a new group")
            self.rows.append(zone)
            y -= unit
        y -= note
        if height_unscaled is None:
            bottom = y - spare
        self.empty = Item("empty", x0, bottom, x1, y)       # below the rows: drop chains here for a new group
        self.frame = Item("frame", frame_x0, bottom, x1, top)

        # The armature pane: its arrow in a strip of its own, then (open) a row per armature.
        self.nav_toggle = Item("nav_toggle", frame_x0, bottom, frame_x0 + gutter, title.y0) if gutter else None
        self.nav_rows = []
        if pane:
            ny = title.y0 - pad
            for armature in armatures:
                if ny - unit < bottom:
                    break
                self.nav_rows.append(Item("nav", frame_x0 + gutter, ny - unit, x0 - pad, ny, root=armature.name,
                                          text=names[armature.name], enabled=armature == obj))
                ny -= unit

        # The edges that resize it, and the scroll bar: checked first, they sit over the rows' ends.
        self.bottom_edge = Item("edge_bottom", x0, bottom, x1, bottom + edge)
        self.right_edge = Item("edge_right", x1 - edge, bottom, x1, title.y0)
        corner = Item("corner", x1 - 3 * edge, bottom, x1, bottom + 3 * edge)
        self.items += [corner, self.bottom_edge, self.right_edge] + self.nav_rows
        if self.nav_toggle is not None:
            self.items.append(self.nav_toggle)
        self.track = self.thumb = None
        self.rows_per_pixel = 0.0
        if self.overflow:
            track_bottom = scroll_bottom
            bar_x0, bar_x1 = x1 - edge - BAR * scale, x1 - edge
            self.track = Item("scroll_track", bar_x0, track_bottom, bar_x1, body_top)
            span = body_top - track_bottom
            length = max(span * room / len(rows), unit * 0.75)
            thumb_top = body_top - (span - length) * self.first / max(len(rows) - room, 1)
            self.thumb = Item("scroll_thumb", bar_x0, thumb_top - length, bar_x1, thumb_top)
            self.rows_per_pixel = (len(rows) - room) / max(span - length, 1.0)
            self.items += [self.thumb, self.track]
        self.items += [close] + buttons + self.rows + [self.empty, title]

    def hit(self, x, y):
        """The item under a point (the most specific one), the frame if nothing else, or None outside."""
        if not self.frame.contains(x, y):
            return None
        for item in self.items:
            if item.contains(x, y):
                return item
        return self.frame

    def boxed(self, y0, y1):
        """The chains of the rows a box spanning these heights touches: a chain's row gives the chain, a
        folder's row every chain of its group (it may be folded)."""
        low, high = min(y0, y1), max(y0, y1)
        found = []
        for row in self.rows:
            if row.y1 > low and row.y0 < high:
                if row.kind == "chain":
                    found.append((row.group, row.root))
                elif row.kind == "group" and self.obj is not None:
                    found += [(row.group, root.name) for root in self.obj.waifu_physics.groups[row.group].roots]
        return list(dict.fromkeys(found))

    def boxed_bones(self, y0, y1):
        """The bones of the bone rows a box spanning these heights touches."""
        low, high = min(y0, y1), max(y0, y1)
        return [row.root for row in self.rows if row.kind == "bone" and row.y1 > low and row.y0 < high]

    def drop_target(self, x, y):
        """Where dropped chains or bones would go: ("group", index), ("new", -1), or None."""
        item = self.hit(x, y)
        if item is None:
            return None
        if item.kind in ("group", "fold", "chain", "chain_fold", "bone", "links", "link", "link_remove"):
            return ("group", item.group)
        if item.kind in ("newzone", "empty", "edge_bottom", "corner"):
            return ("new", -1)
        return None


def pane_names(names):
    """The pane's short names: what the armatures' names share at the front is dropped (VRoid names its rigs
    "WS Rig | <character>'s hair", "... dress"), back to a word boundary."""
    if len(names) < 2:
        return {name: name for name in names}
    import os
    shared = os.path.commonprefix(names)
    cut = max(shared.rfind(" "), shared.rfind("|"), shared.rfind("_"), shared.rfind("."))
    shared = shared[:cut + 1] if cut >= 0 else ""
    return {name: (name[len(shared):].strip() or name) for name in names}


class Drag:
    __slots__ = ("kind", "group", "count", "x", "y")

    def __init__(self, kind, group, count, x, y):
        self.kind, self.group, self.count, self.x, self.y = kind, group, count, x, y


def _scale(context):
    return context.preferences.system.ui_scale or 1.0


def active_armature(context):
    obj = context.object
    return obj if obj is not None and obj.type == "ARMATURE" else None


def listed(context):
    """The armatures the pane lists: every one in the scene with a group, and the active one."""
    found = [obj for obj in context.scene.objects if obj.type == "ARMATURE" and len(obj.waifu_physics.groups)]
    active = active_armature(context)
    if active is not None and active not in found:
        found.insert(0, active)
    return found


def shown(context):
    """The armature the manager shows: the one picked in its pane, else the active armature, else the first
    armature with a group. It need not be selected or active."""
    armatures = listed(context)
    picked = _shown.get(context.area.as_pointer()) if context.area is not None else None
    for obj in armatures:
        if obj.session_uid == picked:
            return obj
    active = active_armature(context)
    return active if active is not None else armatures[0] if armatures else None


def _font(context, scale):
    """The manager's font, at the sidebar's widget size: set before measuring or drawing text."""
    blf.size(0, context.preferences.ui_styles[0].widget.points * scale)


def _measure(text):
    return blf.dimensions(0, text)[0]


def layout_for(context):
    area, region = context.area, context.region
    key = area.as_pointer()
    scale = _scale(context)
    _font(context, scale)
    place = _places.get(key)
    if place is None:
        tools = next((r for r in area.regions if r.type == "TOOLS"), None)
        inset = tools.width if tools is not None and context.preferences.system.use_region_overlap else 0
        place = (inset + 10 * scale, region.height - 110 * scale)       # under the view's name
    obj = shown(context)
    opened = frozenset(root for uid, root in _open_chains if obj is not None and uid == obj.session_uid)
    return Layout(obj, (region.width, region.height), scale, place, _scroll.get(key, 0),
                  _drags.get(key), _sizes.get(key), armatures=listed(context), nav_open=key not in _nav_closed,
                  measure=_measure, open_chains=opened)


def scroll(context, rows):
    key = context.area.as_pointer()
    layout = layout_for(context)
    _scroll[key] = min(max(layout.first + rows, 0), max(layout.total - layout.capacity, 0))
    context.area.tag_redraw()


# --------------------------------------------------------------------------- drawing

def _palette(context):
    ui = context.preferences.themes[0].user_interface
    back, item, tool = ui.wcol_menu_back, ui.wcol_list_item, ui.wcol_tool
    text = tuple(item.text) + (1.0,)
    return {
        "back": tuple(back.inner[:3]) + (0.94,),
        "outline": tuple(back.outline[:3]) + (1.0,),
        "title": tuple(min(c + 0.035, 1.0) for c in back.inner[:3]) + (1.0,),
        "text": text,
        "dim": text[:3] + (0.5,),
        "text_sel": tuple(item.text_sel) + (1.0,),
        "sel": tuple(item.inner_sel[:3]) + (0.9,),
        "hover": text[:3] + (0.07,),
        "target": tuple(item.inner_sel[:3]) + (0.35,),
        "button": tuple(tool.inner[:3]) + (1.0,),
        "button_off": tuple(tool.inner[:3]) + (0.4,),
        "button_text": tuple(tool.text) + (1.0,),
        "danger": (0.62, 0.18, 0.18, 1.0),
        "edge": tuple(item.inner_sel[:3]) + (0.85,),
        "edge_held": tuple(min(c + 0.2, 1.0) for c in item.inner_sel[:3]) + (1.0,),
        "field": tuple(ui.wcol_text.inner[:3]) + (1.0,),
    }


class _Canvas:
    """Collects the frame's shapes, so they draw in two batches whatever the row count."""

    def __init__(self, scale):
        self.scale, self.tris, self.tri_colours, self.lines, self.line_colours = scale, [], [], [], []

    def rect(self, x0, y0, x1, y1, colour, radius=0.0):
        if radius <= 0:
            self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], colour)
            return
        import math
        r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
        points = []
        for cx, cy, start in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180),
                              (x1 - r, y0 + r, 270)):
            for step in range(4):
                a = math.radians(start + step * 30)
                points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        self.poly(points, colour)

    def poly(self, points, colour):
        """A convex polygon, as a fan."""
        for k in range(1, len(points) - 1):
            self.tris += [points[0], points[k], points[k + 1]]
            self.tri_colours += [colour] * 3

    def line(self, points, colour):
        for a, b in zip(points, points[1:]):
            self.lines += [a, b]
            self.line_colours += [colour, colour]

    def outline(self, x0, y0, x1, y1, colour):
        self.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], colour)

    def draw(self, region):
        gpu.state.blend_set("ALPHA")
        if self.tris:
            shader = gpu.shader.from_builtin("FLAT_COLOR")
            batch_for_shader(shader, "TRIS", {"pos": self.tris, "color": self.tri_colours}).draw(shader)
        if self.lines:
            shader = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
            shader.uniform_float("viewportSize", (region.width, region.height))
            shader.uniform_float("lineWidth", 1.2 * self.scale)
            batch_for_shader(shader, "LINES", {"pos": self.lines, "color": self.line_colours}).draw(shader)


def _arrow(canvas, x, y, s, open_, colour):
    if open_:
        canvas.poly([(x - 4 * s, y + 2.5 * s), (x + 4 * s, y + 2.5 * s), (x, y - 3 * s)], colour)
    else:
        canvas.poly([(x - 2 * s, y - 4 * s), (x - 2 * s, y + 4 * s), (x + 3.5 * s, y)], colour)


def _arrow_left(canvas, x, y, s, colour):
    canvas.poly([(x + 2 * s, y - 4 * s), (x - 3.5 * s, y), (x + 2 * s, y + 4 * s)], colour)


def _folder(canvas, x, y, s, colour):
    canvas.rect(x, y - 4.5 * s, x + 13 * s, y + 3.5 * s, colour, radius=1.2 * s)
    canvas.rect(x, y + 2.5 * s, x + 5.5 * s, y + 5 * s, colour, radius=1 * s)


def _chain_icon(canvas, x, y, s, colour):
    """A chain: three joints on a zig-zag."""
    joints = [(x + 1 * s, y + 4 * s), (x + 5 * s, y - 1 * s), (x + 9 * s, y + 3 * s)]
    canvas.line(joints, colour)
    for jx, jy in joints:
        canvas.rect(jx - 1.6 * s, jy - 1.6 * s, jx + 1.6 * s, jy + 1.6 * s, colour, radius=1.6 * s)


def _bone_icon(canvas, x, y, s, colour):
    """A bone, as Blender draws one: a long diamond from its head."""
    canvas.poly([(x, y), (x + 3 * s, y + 2.6 * s), (x + 10 * s, y), (x + 3 * s, y - 2.6 * s)], colour)


def _link_icon(canvas, x, y, s, colour):
    """A link: two rings joined."""
    for cx in (x + 2 * s, x + 8 * s):
        canvas.outline(cx - 2 * s, y - 2 * s, cx + 2 * s, y + 2 * s, colour)
    canvas.line([(x + 4 * s, y), (x + 6 * s, y)], colour)


def _cross(canvas, x, y, s, colour):
    canvas.line([(x - 4 * s, y - 4 * s), (x + 4 * s, y + 4 * s)], colour)
    canvas.line([(x - 4 * s, y + 4 * s), (x + 4 * s, y - 4 * s)], colour)


def _fit(text, room):
    if blf.dimensions(0, text)[0] <= room:
        return text
    while text and blf.dimensions(0, text + "…")[0] > room:
        text = text[:-1]
    return text + "…"


def draw(context):
    area, region = context.area, context.region
    key = area.as_pointer()
    layout = layout_for(context)
    s, unit = layout.scale, layout.unit
    colours = _palette(context)
    canvas = _Canvas(s)
    texts = []                                  # (x, y, text, colour), drawn over the shapes

    def label(item, text, x, colour, room=None, right=False):
        width = blf.dimensions(0, text)[0]
        if room is not None:
            text = _fit(text, room)
            width = blf.dimensions(0, text)[0]
        texts.append((x - width if right else x, (item.y0 + item.y1) / 2 - text_mid, text, colour))

    _font(context, s)
    text_mid = blf.dimensions(0, "Xg")[1] * 0.38
    frame, title = layout.frame, layout.title
    canvas.rect(frame.x0, frame.y0, frame.x1, frame.y1, colours["back"], radius=5 * s)
    canvas.rect(title.x0, title.y0, title.x1, title.y1, colours["title"], radius=5 * s)
    canvas.rect(title.x0, title.y0, title.x1, title.y0 + 5 * s, colours["title"])
    hover = _hover.get(key)
    hovered = layout.hit(*hover) if hover is not None else None
    drag = _drags.get(key)
    target = layout.drop_target(drag.x, drag.y) if drag is not None else None
    if drag is not None and drag.kind == "group" and target == ("new", -1):
        target = None                                   # a group dropped on empty space stays as it is
    if drag is not None and target is not None and target[0] == "group" and target[1] == drag.group:
        target = None                                   # onto itself

    label(title, title.text, title.x0 + 8 * s, colours["text"], room=title.x1 - title.x0 - unit - 12 * s)
    toggle_item = layout.nav_toggle
    if toggle_item is not None:                                  # the armature pane: its arrow, then its names
        if hovered is toggle_item:
            canvas.rect(toggle_item.x0 + 2 * s, frame.y0 + 2 * s, toggle_item.x1 - 1 * s, title.y0 - 2 * s,
                        colours["hover"], radius=3 * s)
        arrow_x, arrow_y = (toggle_item.x0 + toggle_item.x1) / 2 + 1 * s, title.y0 - 4 * s - unit / 2
        lit = colours["text"] if hovered is toggle_item else colours["dim"]
        if layout.nav_open:
            _arrow_left(canvas, arrow_x, arrow_y, s, lit)
        else:
            _arrow(canvas, arrow_x, arrow_y, s, False, lit)
    if layout.nav_open:
        canvas.rect(toggle_item.x1, frame.y0 + 2 * s, layout.nav_x1 - 1 * s, title.y0 - 2 * s,
                    colours["title"], radius=3 * s)
        for row in layout.nav_rows:
            if row.enabled:
                canvas.rect(row.x0, row.y0 + 1, row.x1, row.y1 - 1, colours["sel"], radius=3 * s)
            elif hovered is row:
                canvas.rect(row.x0, row.y0 + 1, row.x1, row.y1 - 1, colours["hover"], radius=3 * s)
            label(row, row.text, row.x0 + 5 * s, colours["text_sel"] if row.enabled else colours["text"],
                  room=row.x1 - row.x0 - 8 * s)
    close = next(i for i in layout.items if i.kind == "close")
    _cross(canvas, (close.x0 + close.x1) / 2, (close.y0 + close.y1) / 2, s,
           colours["text"] if hovered is close else colours["dim"])

    for item in layout.items:
        if item.kind in ("new", "merge", "delete"):
            fill = colours["button"] if item.enabled else colours["button_off"]
            if item.kind == "delete" and item.enabled:
                fill = colours["danger"]
            if item.enabled and hovered is item:
                fill = tuple(min(c + 0.06, 1.0) for c in fill[:3]) + (1.0,)
            canvas.rect(item.x0, item.y0, item.x1, item.y1, fill, radius=3 * s)
            text_colour = colours["button_text"] if item.enabled else colours["dim"]
            width = blf.dimensions(0, item.text)[0]
            label(item, item.text, (item.x0 + item.x1 - width) / 2, text_colour)

    obj = layout.obj
    if obj is None:
        texts.append((frame.x0 + 10 * s, title.y0 - unit, "Make an armature active to see its groups.", colours["dim"]))
    elif not len(obj.waifu_physics.groups):
        texts.append((frame.x0 + 10 * s, title.y0 - 2.4 * unit, "No groups yet. Select bones in Pose Mode,",
                      colours["dim"]))
        texts.append((frame.x0 + 10 * s, title.y0 - 3.2 * unit, "then click + in the sidebar's Groups.",
                      colours["dim"]))

    active = obj.waifu_physics.active_group if obj is not None else -1
    box = _boxes.get(key)
    boxed = set(layout.boxed(box[1], box[3])) if box is not None else set()
    boxed_bones = set(layout.boxed_bones(box[1], box[3])) if box is not None else set()
    opened = layout.open_chains
    for row in layout.rows:
        mid = (row.y0 + row.y1) / 2
        if row.kind == "newzone":
            lit = target == ("new", -1)
            if lit:
                canvas.rect(row.x0, row.y0 + 1, row.x1, row.y1 - 1, colours["target"], radius=3 * s)
            canvas.outline(row.x0, row.y0 + 1, row.x1, row.y1 - 1, colours["text"] if lit else colours["dim"])
            width = blf.dimensions(0, row.text)[0]
            label(row, row.text, (row.x0 + row.x1 - width) / 2, colours["text"] if lit else colours["dim"])
            continue
        partly = False
        if row.kind == "chain":
            picked = row.state == "all" or (row.group, row.root) in boxed
            partly = not picked and row.state == "some"
        elif row.kind == "bone":
            picked = row.root in layout.selected_bones or row.root in boxed_bones
        elif row.kind == "link":
            picked = row.group == active and obj.waifu_physics.groups[row.group].active_link == row.index
        else:
            picked = False
        if target == ("group", row.group):
            canvas.rect(row.x0 + 2 * s, row.y0, row.x1 - 2 * s, row.y1, colours["target"])
        if picked:
            canvas.rect(row.x0 + 2 * s, row.y0 + 1, row.x1 - 2 * s, row.y1 - 1, colours["sel"], radius=3 * s)
        elif partly:                                   # some of its bones selected: a lighter highlight
            canvas.rect(row.x0 + 2 * s, row.y0 + 1, row.x1 - 2 * s, row.y1 - 1, colours["target"], radius=3 * s)
        elif hovered is not None and hovered.y0 == row.y0 and drag is None:
            canvas.rect(row.x0 + 2 * s, row.y0 + 1, row.x1 - 2 * s, row.y1 - 1, colours["hover"], radius=3 * s)
        text_colour = colours["text_sel"] if picked else colours["text"]
        count_right = row.x1 - 8 * s
        if row.kind == "group":
            group = obj.waifu_physics.groups[row.group]
            _arrow(canvas, row.x0 + 12 * s, mid, s, group.show_chains, colours["dim"] if hovered is None
                   or hovered.kind != "fold" or hovered.group != row.group else colours["text"])
            is_active = row.group == active
            _folder(canvas, row.x0 + 22 * s, mid, s, colours["sel"] if is_active else colours["dim"])
            editing = _editing.get(key)
            if editing is not None and editing.group == row.group:     # the name being typed, in a field
                x0, x1 = row.x0 + 37 * s, row.x1 - 30 * s
                canvas.rect(x0, row.y0 + 2 * s, x1, row.y1 - 2 * s, colours["field"], radius=3 * s)
                canvas.outline(x0, row.y0 + 2 * s, x1, row.y1 - 2 * s, colours["edge"])
                shown_text = _fit(editing.text, x1 - x0 - 12 * s) if editing.text else ""
                width = blf.dimensions(0, shown_text)[0]
                if editing.selected and shown_text:
                    canvas.rect(x0 + 4 * s, row.y0 + 4 * s, x0 + 6 * s + width, row.y1 - 4 * s, colours["sel"])
                else:
                    canvas.rect(x0 + 5 * s + width, row.y0 + 5 * s, x0 + 6 * s + width, row.y1 - 5 * s,
                                colours["text"])
                label(row, shown_text, x0 + 5 * s, colours["text_sel"] if editing.selected else colours["text"])
            else:
                name = row.text if group.enabled else row.text + "  (off)"
                label(row, name, row.x0 + 41 * s, colours["text"] if is_active or group.enabled else colours["dim"],
                      room=row.x1 - row.x0 - 80 * s)
            label(row, row.count, count_right, colours["dim"], right=True)
        elif row.kind == "chain":
            lit = hovered is not None and hovered.kind == "chain_fold" and hovered.root == row.root
            _arrow(canvas, row.x0 + 28 * s, mid, s, row.root in opened, colours["text"] if lit else colours["dim"])
            _chain_icon(canvas, row.x0 + 38 * s, mid, s, text_colour if picked else colours["dim"])
            label(row, row.text, row.x0 + 54 * s, text_colour, room=row.x1 - row.x0 - 88 * s)
            label(row, row.count, count_right, colours["dim"] if not picked else text_colour, right=True)
        elif row.kind == "bone":
            _bone_icon(canvas, row.x0 + 54 * s, mid, s, text_colour if picked else colours["dim"])
            label(row, row.text, row.x0 + 68 * s, text_colour, room=row.x1 - row.x0 - 108 * s)
            label(row, row.count, count_right, colours["dim"] if not picked else text_colour, right=True)
        elif row.kind == "links":
            group = obj.waifu_physics.groups[row.group]
            lit = hovered is not None and hovered.y0 == row.y0
            _arrow(canvas, row.x0 + 28 * s, mid, s, group.list_links, colours["text"] if lit else colours["dim"])
            _link_icon(canvas, row.x0 + 38 * s, mid, s, colours["dim"])
            label(row, row.text, row.x0 + 54 * s, colours["dim"])
            label(row, row.count, count_right, colours["dim"], right=True)
        else:                                          # a link: its two bones, and an x to remove it
            label(row, row.text, row.x0 + 68 * s, text_colour, room=row.x1 - row.x0 - 93 * s)
            if hovered is not None and hovered.y0 == row.y0 and drag is None:
                over = hovered.kind == "link_remove"
                _cross(canvas, row.x1 - unit / 2, mid, s * 0.8,
                       colours["text"] if over else colours["dim"] if not picked else text_colour)

    held = _pressed.get(key)
    if layout.track is not None:                                # the scroll bar
        track, thumb = layout.track, layout.thumb
        canvas.rect(track.x0, track.y0, track.x1, track.y1, colours["text"][:3] + (0.06,), radius=4 * s)
        lit = held == "scroll_thumb" or (hovered is thumb and drag is None)
        canvas.rect(thumb.x0 + 1 * s, thumb.y0 + 1 * s, thumb.x1 - 1 * s, thumb.y1 - 1 * s,
                    colours["text"][:3] + (0.65 if held == "scroll_thumb" else 0.45 if lit else 0.28,),
                    radius=4 * s)

    # The resizable edges show a thin bar under the mouse, brighter while held.
    grabbed = held if held in ("edge_bottom", "edge_right", "corner") else None
    over = hovered.kind if hovered is not None and drag is None and hovered.kind in (
        "edge_bottom", "edge_right", "corner") else None
    for kind in {grabbed or over} - {None}:
        colour = colours["edge_held"] if grabbed else colours["edge"]
        if kind in ("edge_bottom", "corner"):
            canvas.rect(frame.x0 + 5 * s, frame.y0, frame.x1 - 5 * s, frame.y0 + 3 * s, colour, radius=1.5 * s)
        if kind in ("edge_right", "corner"):
            canvas.rect(frame.x1 - 3 * s, frame.y0 + 5 * s, frame.x1, title.y0 - 2 * s, colour, radius=1.5 * s)

    if box is not None:                                         # the box select
        x0, x1 = sorted((box[0], box[2]))
        y0, y1 = sorted((box[1], box[3]))
        x0, x1 = max(x0, frame.x0), min(x1, frame.x1)
        y0, y1 = max(y0, frame.y0), min(y1, frame.y1)
        canvas.rect(x0, y0, x1, y1, colours["sel"][:3] + (0.15,))
        canvas.outline(x0, y0, x1, y1, colours["text"][:3] + (0.8,))      # visible over selected rows too

    if drag is not None:                                        # what is being dragged, by the mouse
        text = (f"{drag.count} chain{'' if drag.count == 1 else 's'}" if drag.kind == "chains"
                else f"{drag.count} bone{'' if drag.count == 1 else 's'}" if drag.kind == "bones"
                else f"Merge '{obj.waifu_physics.groups[drag.group].name}'" if obj is not None else "")
        if drag.kind == "group" and target is not None:
            text += f" into '{obj.waifu_physics.groups[target[1]].name}'"
        width = blf.dimensions(0, text)[0]
        x, y = drag.x + 14 * s, drag.y - 20 * s
        canvas.rect(x - 6 * s, y - 6 * s, x + width + 6 * s, y + 14 * s, colours["sel"][:3] + (1.0,), radius=3 * s)
        texts.append((x, y, text, colours["text_sel"]))

    canvas.draw(region)
    for x, y, text, colour in texts:
        blf.position(0, x, y, 0)
        blf.color(0, *colour)
        blf.draw(0, text)
    gpu.state.blend_set("NONE")


# --------------------------------------------------------------------------- input: the gizmo over the manager

def _call(name, **properties):
    """Run one of the add-on's operators on the armature the manager shows (which need not be the active object),
    as one undo step (from Python, bpy.ops pushes none unless asked)."""
    operator = getattr(bpy.ops.waifu_physics, name)
    obj = shown(bpy.context)
    if obj is None:
        return
    with bpy.context.temp_override(object=obj, active_object=obj):
        operator("EXEC_DEFAULT", True, **properties)


class WAIFU_PHYSICS_GT_chain_manager(bpy.types.Gizmo):
    bl_idname = "WAIFU_PHYSICS_GT_chain_manager"

    def draw(self, context):
        draw(context)

    def test_select(self, context, location):
        key = context.area.as_pointer()
        inside = layout_for(context).hit(*location) is not None
        before = _hover.get(key)
        if inside:
            _hover[key] = tuple(location)
        else:
            _hover.pop(key, None)
        if before != _hover.get(key):
            context.area.tag_redraw()
        return 0 if inside else -1

    def invoke(self, context, event):
        x, y = event.mouse_region_x, event.mouse_region_y
        layout = layout_for(context)
        item = layout.hit(x, y)
        self.press, self.item, self.dragging, self.pending, self.box = (x, y), item, False, None, None
        self.origin = (layout.frame.x0, layout.title.y1)
        self.size = (layout.main_width, (layout.frame.y1 - layout.frame.y0) / layout.scale)
        self.first = layout.first
        if item is None:
            return {"FINISHED"}
        obj, kind = layout.obj, item.kind
        _pressed[context.area.as_pointer()] = kind
        if kind == "scroll_track":                  # above or below the thumb: a page
            page = layout.capacity - 1
            scroll(context, -page if y > layout.thumb.y1 else page)
        elif kind == "close":
            toggle(context.area)
        elif kind == "nav_toggle":
            _nav_closed.symmetric_difference_update({context.area.as_pointer()})
        elif kind == "nav":
            found = bpy.data.objects.get(item.root)
            if found is not None:
                _shown[context.area.as_pointer()] = found.session_uid
                _scroll[context.area.as_pointer()] = 0
        elif kind in ("new", "merge", "delete") and item.enabled:
            if kind == "new":
                _call("chains_to_group", index=-1)
            elif kind == "merge":
                _call("groups_merge")
            else:
                _call("chains_remove")
        elif kind == "fold":
            group = obj.waifu_physics.groups[item.group]
            group.show_chains = not group.show_chains
        elif kind == "chain_fold":
            _open_chains.symmetric_difference_update({(obj.session_uid, item.root)})
        elif kind == "links":
            group = obj.waifu_physics.groups[item.group]
            group.list_links = not group.list_links
        elif kind == "link":
            _call("link_pick", group=item.group, index=item.index)
        elif kind == "link_remove":
            _call("link_remove", group=item.group, index=item.index)
        elif kind == "bone":
            if event.shift:
                _call("bone_click", group=item.group, root=item.chain, bone=item.root, span=True)
            elif event.ctrl:
                _call("bone_click", group=item.group, root=item.chain, bone=item.root, extend=True)
            elif item.root in layout.selected_bones:
                self.pending = item          # released without a drag: select just this one
            else:
                _call("bone_click", group=item.group, root=item.chain, bone=item.root)
        elif kind == "chain":
            picked = (item.group, item.root) in layout.chosen
            if event.shift:
                _call("chain_click", group=item.group, root=item.root, span=True)
            elif event.ctrl:
                _call("chain_click", group=item.group, root=item.root, extend=True)
            elif picked:
                self.pending = item          # released without a drag: select just this one
            else:
                _call("chain_click", group=item.group, root=item.root)
        elif kind == "group":
            import time
            key = context.area.as_pointer()
            now, last = time.monotonic(), _last_group_press.get(key)
            _last_group_press[key] = (item.group, now)
            window = context.preferences.inputs.mouse_double_click_time / 1000.0
            if last is not None and last[0] == item.group and now - last[1] <= window and not (event.ctrl or event.shift):
                # A double-click renames. Detected here: the gizmo takes the first press, so the second may
                # never reach a keymap as a double-click.
                _last_group_press.pop(key, None)
                _pressed.pop(key, None)
                bpy.ops.waifu_physics.group_rename("INVOKE_DEFAULT", index=item.group)
                return {"FINISHED"}
            _call("group_click", index=item.group, extend=event.ctrl or event.shift)
        elif kind in ("empty", "frame") and obj is not None:
            self.box = event.shift or event.ctrl       # a box select, adding with Shift or Ctrl; a click selects none
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event, tweak):
        key = context.area.as_pointer()
        if event.type in ("WHEELUPMOUSE", "WHEELDOWNMOUSE"):
            scroll(context, -2 if event.type == "WHEELUPMOUSE" else 2)
            return {"RUNNING_MODAL"}
        if event.type != "MOUSEMOVE" or self.item is None:
            return {"RUNNING_MODAL"}
        x, y = event.mouse_region_x, event.mouse_region_y
        dx, dy = x - self.press[0], y - self.press[1]
        kind = self.item.kind
        if kind == "title":
            _places[key] = (self.origin[0] + dx, self.origin[1] + dy)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if kind in ("edge_bottom", "edge_right", "corner"):
            s = _scale(context)
            width, height = _sizes.get(key, (self.size[0], None))
            if kind in ("edge_right", "corner"):
                width = min(max(self.size[0] + dx / s, MIN_WIDTH), MAX_WIDTH)
            if kind in ("edge_bottom", "corner"):
                height = max(self.size[1] - dy / s, MIN_HEIGHT)
            _sizes[key] = (width, height)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if kind == "scroll_thumb":
            layout = layout_for(context)
            _scroll[key] = self.first + round(-dy * layout.rows_per_pixel)
            scroll(context, 0)                      # clamped
            return {"RUNNING_MODAL"}
        if self.box is not None:
            if key in _boxes or dx * dx + dy * dy > (DRAG_START * _scale(context)) ** 2:
                _boxes[key] = (self.press[0], self.press[1], x, y)
                context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if not self.dragging and self.item.kind in ("chain", "group", "bone") \
                and dx * dx + dy * dy > (DRAG_START * _scale(context)) ** 2:
            layout = layout_for(context)
            if self.item.kind == "chain":
                if (self.item.group, self.item.root) not in layout.chosen:
                    return {"RUNNING_MODAL"}                  # Ctrl-click dropped it: nothing to drag
                _drags[key] = Drag("chains", self.item.group, len(layout.chosen), x, y)
            elif self.item.kind == "bone":
                if self.item.root not in layout.selected_bones:
                    return {"RUNNING_MODAL"}
                simulated = {name for row in layout.rows if row.kind == "bone" for name in [row.root]}
                count = len(layout.selected_bones & simulated) or 1
                _drags[key] = Drag("bones", self.item.group, count, x, y)
            else:
                _drags[key] = Drag("group", self.item.group, 1, x, y)
            self.dragging, self.pending = True, None
        if self.dragging:
            drag = _drags[key]
            drag.x, drag.y = x, y
            _hover[key] = (x, y)
            context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def exit(self, context, cancel):
        key = context.area.as_pointer()
        _pressed.pop(key, None)
        box = _boxes.pop(key, None)
        if self.box is not None and not cancel:
            if box is not None:
                layout = layout_for(context)
                chains = layout.boxed(box[1], box[3])
                _call("chains_set", chains="\n".join(f"{group}|{root}" for group, root in chains),
                      bones="\n".join(layout.boxed_bones(box[1], box[3])), extend=self.box)
            elif not self.box:
                _call("chains_select", action="NONE")
            self.box = None
            context.area.tag_redraw()
            return
        drag = _drags.pop(key, None)
        if drag is not None and not cancel:
            target = layout_for(context).drop_target(drag.x, drag.y)
            obj = shown(context)
            if drag.kind in ("chains", "bones") and target is not None:
                chosen = ops.selected_chain_keys(obj)
                if target[0] == "new" or any(group != target[1] for group, _root in chosen):
                    _call("chains_to_group", index=target[1])
            elif drag.kind == "group" and target is not None and target[0] == "group" \
                    and target[1] != drag.group:
                _call("groups_merge", source=drag.group, target=target[1])
        elif self.pending is not None and not cancel:
            if self.pending.kind == "bone":
                _call("bone_click", group=self.pending.group, root=self.pending.chain, bone=self.pending.root)
            else:
                _call("chain_click", group=self.pending.group, root=self.pending.root)
        self.pending = None
        context.area.tag_redraw()


class WAIFU_PHYSICS_GGT_chain_manager(bpy.types.GizmoGroup):
    bl_idname = "WAIFU_PHYSICS_GGT_chain_manager"
    bl_label = "Waifu Physics Chain Manager"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"PERSISTENT", "SHOW_MODAL_ALL"}

    @classmethod
    def poll(cls, context):
        return is_open(context.area)

    @classmethod
    def setup_keymap(cls, keyconfig):
        keymap = keyconfig.keymaps.new(name="Waifu Physics Chain Manager", space_type="EMPTY", region_type="WINDOW")
        items = keymap.keymap_items
        items.new("gizmogroup.gizmo_tweak", "LEFTMOUSE", "PRESS", any=True)
        items.new("waifu_physics.group_rename", "LEFTMOUSE", "DOUBLE_CLICK")
        items.new("waifu_physics.group_rename", "F2", "PRESS")
        items.new("waifu_physics.manager_scroll", "WHEELUPMOUSE", "PRESS", any=True).properties.rows = -2
        items.new("waifu_physics.manager_scroll", "WHEELDOWNMOUSE", "PRESS", any=True).properties.rows = 2
        items.new("waifu_physics.manager_scroll", "TRACKPADPAN", "ANY", any=True)
        items.new("waifu_physics.manager_key", "X", "PRESS").properties.action = "DELETE"
        items.new("waifu_physics.manager_key", "DEL", "PRESS").properties.action = "DELETE"
        items.new("waifu_physics.manager_key", "A", "PRESS").properties.action = "ALL"
        items.new("waifu_physics.manager_key", "A", "PRESS", alt=True).properties.action = "NONE"
        return keymap

    def setup(self, context):
        gizmo = self.gizmos.new(WAIFU_PHYSICS_GT_chain_manager.bl_idname)
        gizmo.use_draw_modal = True               # drawn while its own drag is on, too


# --------------------------------------------------------------------------- operators

class WAIFU_PHYSICS_OT_chain_manager(bpy.types.Operator):
    bl_idname = "waifu_physics.chain_manager"
    bl_label = "Chain Manager"
    bl_description = ("Show or hide the chain manager: organize chains into groups")

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def execute(self, context):
        toggle(context.area)
        return {"FINISHED"}


class WAIFU_PHYSICS_OT_manager_scroll(bpy.types.Operator):
    bl_idname = "waifu_physics.manager_scroll"
    bl_label = "Scroll Chain Manager"
    bl_options = {"INTERNAL"}

    rows: bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return is_open(context.area) and context.region is not None and context.region.type == "WINDOW"

    def invoke(self, context, event):
        if event.type == "TRACKPADPAN":             # a touchpad's two-finger scroll: pixels, not steps
            key = context.area.as_pointer()
            moved = _pan.get(key, 0.0) + (event.mouse_y - event.mouse_prev_y) / (ROW * _scale(context))
            self.rows = int(moved)
            _pan[key] = moved - self.rows
            if not self.rows:
                return {"FINISHED"}
        return self.execute(context)

    def execute(self, context):
        scroll(context, self.rows)
        return {"FINISHED"}


class WAIFU_PHYSICS_OT_manager_key(bpy.types.Operator):
    bl_idname = "waifu_physics.manager_key"
    bl_label = "Chain Manager Key"
    bl_options = {"UNDO", "INTERNAL"}

    action: bpy.props.EnumProperty(items=(("DELETE", "Delete", ""), ("ALL", "All", ""), ("NONE", "None", "")))

    @classmethod
    def poll(cls, context):
        return is_open(context.area) and shown(context) is not None

    def execute(self, context):
        if self.action == "DELETE":
            _call("chains_remove")
        else:
            _call("chains_select", action=self.action)
        context.area.tag_redraw()
        return {"FINISHED"}


class Editing:
    """A group's name being typed in its row: the text, and whether it is all selected (typing replaces it)."""
    __slots__ = ("group", "text", "selected")

    def __init__(self, group, text):
        self.group, self.text, self.selected = group, text, True


class WAIFU_PHYSICS_OT_group_rename(bpy.types.Operator):
    """Edit a group's name in its row: double-click it, or F2 over it. Enter renames, Esc keeps the old name,
    a click elsewhere renames too."""
    bl_idname = "waifu_physics.group_rename"
    bl_label = "Rename Group"
    bl_description = "Rename a group"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(default=-1, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.area is not None and shown(context) is not None

    def invoke(self, context, event):
        obj = shown(context)
        key = context.area.as_pointer()
        if self.index < 0:
            if not is_open(context.area) or context.region is None or context.region.type != "WINDOW":
                return {"PASS_THROUGH"}
            item = layout_for(context).hit(event.mouse_region_x, event.mouse_region_y)
            if item is None or item.kind != "group":
                return {"PASS_THROUGH"}           # not over a group's row: the viewport's own double-click
            self.index = item.group
        if not 0 <= self.index < len(obj.waifu_physics.groups) or key in _editing:
            return {"CANCELLED"}
        _editing[key] = Editing(self.index, obj.waifu_physics.groups[self.index].name)
        self.area_key = context.area.as_pointer()     # not the area itself: it can be closed while typing
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _finish(self, context, keep):
        key = self.area_key
        editing = _editing.pop(key, None)
        _redraw(context, key)
        obj = shown(context)
        name = editing.text.strip() if editing is not None else ""
        if not keep or obj is None or not name or not 0 <= editing.group < len(obj.waifu_physics.groups):
            return False
        obj.waifu_physics.groups[editing.group].name = name
        return True

    def modal(self, context, event):
        editing = _editing.get(self.area_key)
        if editing is None:
            return {"CANCELLED"}
        if not _redraw(context, self.area_key, tag=False):       # its area was closed: drop the edit
            _editing.pop(self.area_key, None)
            return {"CANCELLED"}
        if event.value != "PRESS":
            return {"RUNNING_MODAL"} if event.type not in ("MOUSEMOVE", "INBETWEEN_MOUSEMOVE", "TIMER") \
                else {"PASS_THROUGH"}
        if event.type in ("RET", "NUMPAD_ENTER"):
            return {"FINISHED"} if self._finish(context, True) else {"CANCELLED"}
        if event.type == "ESC":
            self._finish(context, False)
            return {"CANCELLED"}
        if event.type in ("LEFTMOUSE", "RIGHTMOUSE", "MIDDLEMOUSE"):
            renamed = self._finish(context, True)       # a click elsewhere keeps what was typed, as Blender's fields
            return ({"FINISHED"} if renamed else {"CANCELLED"}) | {"PASS_THROUGH"}
        if event.type == "BACK_SPACE":
            editing.text = "" if editing.selected or event.ctrl else editing.text[:-1]
            editing.selected = False
        elif event.type == "A" and event.ctrl:
            editing.selected = True
        elif event.unicode and not (event.ctrl or event.alt or event.oskey):
            editing.text = event.unicode if editing.selected else editing.text + event.unicode
            editing.selected = False
        _redraw(context, self.area_key)
        return {"RUNNING_MODAL"}


def _redraw(context, key, tag=True):
    """Redraw the area whose pointer is key, found among the open windows' areas. Returns whether it is open."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.as_pointer() == key:
                if tag:
                    area.tag_redraw()
                return True
    return False


CLASSES = (WAIFU_PHYSICS_OT_chain_manager, WAIFU_PHYSICS_OT_manager_scroll, WAIFU_PHYSICS_OT_manager_key,
           WAIFU_PHYSICS_OT_group_rename, WAIFU_PHYSICS_GT_chain_manager,
           WAIFU_PHYSICS_GGT_chain_manager)


@bpy.app.handlers.persistent
def _file_loaded(_dummy):
    """A new file brings new areas: the manager starts closed."""
    for state in (_open, _places, _scroll, _hover, _drags, _sizes, _pressed, _pan, _boxes, _shown, _nav_closed,
                  _last_group_press, _editing, _open_chains):
        state.clear()


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.app.handlers.load_post.append(_file_loaded)
    # A still double-click (the mouse not moved since the first click ended) reaches no gizmo, so the viewport's
    # keymap has it too; over anything but a group's row the operator passes it on.
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is not None:
        keymap = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
        for event, value in (("LEFTMOUSE", "DOUBLE_CLICK"), ("F2", "PRESS")):
            _addon_keymaps.append((keymap, keymap.keymap_items.new("waifu_physics.group_rename", event, value)))


def unregister():
    for keymap, item in _addon_keymaps:
        try:
            keymap.keymap_items.remove(item)
        except (ReferenceError, RuntimeError):
            pass
    _addon_keymaps.clear()
    if _file_loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_file_loaded)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    _file_loaded(None)
