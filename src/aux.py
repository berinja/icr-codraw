#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auxiliary functions and classes to build and analyse games and dialogues.
"""

from collections import namedtuple, Counter
import matplotlib.pyplot as plt
import numpy as np

from cliparts import file2obj
from codraw_data import AbstractScene
from abs_metric import scene_similarity

N_OBJ_ATTRIBUTES = 8
NON_EXISTING = -10000

Turn = namedtuple("Turn", ["teller", "drawer"])
EditTrack = namedtuple('EditTrack', ['added_at', 'edits', 'edit_turns'])

labels_dic = {'not_cr': 0, 'cr': 1}


class ClipArtObj:
    """
    Represent a ClipArt with position (x, y), flip, size and whether it
    exists in a scene.
    """
    def __init__(self, attributes):
        (self.png, self.local_idx, self.clipart_obj, self.clipart_type,
            self.x, self.y, self.z, self.flip) = attributes
        self.exists = self.is_shown()

    def is_shown(self):
        """Return True if the clipart is shown in the scene."""
        # according to
        # https://github.com/facebookresearch/CoDraw/blob/b209770a327f48fdd768724bbcf2783897b0c7fb/js/Abs_util.js#L80
        if (int(float(self.x)) != NON_EXISTING
                and int(float(self.y)) != NON_EXISTING):
            return True
        return False

    @property
    def name(self):
        return file2obj[self.png]


def get_objects(symbolic_scene):
    """Return a list of objects (ClipArtObj) in a scene."""
    n_objects, *objs = symbolic_scene.split(',')
    objects = [objs[i: i+N_OBJ_ATTRIBUTES] for i in range(0, len(objs), N_OBJ_ATTRIBUTES)]
    objects = [ClipArtObj(attributes) for attributes in objects]
    return objects


def list_objects(symbolic_scene):
    """Prints the objects in a scene."""
    objects = get_objects(symbolic_scene)
    for obj in objects:
        print(f'{obj.name}: {obj.x}, {obj.y}, {obj.z}, {obj.flip}')


def object_positions(objects):
    """Return a list of (id, x, y) positions."""
    return [(obj.png, obj.x, obj.y) for obj in objects]


def scene_is_empty(symbolic_scene):
    """Check whether a scene has no cliparts."""
    objects = get_objects(symbolic_scene)
    positions = object_positions(objects)
    for _, x, y in positions:
        # apparently some empty images have a few objects with one
        # coordinate = -10000 and the other valid...
        # we consider cliparts with both coordinates valid, according to
        # https://github.com/facebookresearch/CoDraw/blob/b209770a327f48fdd768724bbcf2783897b0c7fb/js/Abs_util.js#L80
        if int(float(x)) != NON_EXISTING and int(float(y)) != NON_EXISTING:
            return False
    return True


def n_present_objects(symbolic_scene):
    """Return the number of objects present in a scene."""
    return len([x for x in get_objects(symbolic_scene) if x.exists])


class Action():
    """Represents an action."""
    def __init__(self, clipart, action):
        self.clipart = clipart
        self.action = action

    def print_(self):
        print(f'  {self.action} {self.clipart}')

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return self.clipart == other.clipart and self.action == other.action


def drawer_actions(scene_before, scene_after, verbose=False):
    """Return a list with the drawer's actions in a dialogue turn."""

    actions = []

    before_is_empty = scene_is_empty(scene_before)
    after_is_empty = scene_is_empty(scene_after)
    objects_before = get_objects(scene_before) if not before_is_empty else []
    objects_after = get_objects(scene_after) if not after_is_empty else []

    # Case 1: nothing going on, both scenes are empty
    if before_is_empty and after_is_empty:
        if verbose:
            print(' empty scenes')

    # Case 2: scene gets empty, this case should never occur
    elif not before_is_empty and after_is_empty:
        print(' CHECK ME, why is only the after scene empty?')

    # Case 3: emtpy scene is initialized, thus all objects are newly added
    elif before_is_empty and not after_is_empty:
        # TODO: do we care if added objects were flipped and changed size?
        found = False
        for obj in objects_after:
            if obj.exists:
                found = True
                act = Action(obj.name, 'added')
                actions.append(act)
        # ensure that at least one object was indeed added
        assert found

    # Case 4: scene is altered
    elif not before_is_empty and not after_is_empty:
        # TODO: is it ok to assume that each clipart occurs only once always?
        objects_before = {obj.name: obj for obj in objects_before}
        objects_after = {obj.name: obj for obj in objects_after}

        # loop over all objects in the current scene and compare their status
        # with the previous scene
        for name, obj in objects_after.items():
            obj_before = objects_before[name]

            # Case a: object was deleted
            if not obj.exists and obj_before.exists:
                act = Action(name, 'deleted')
                actions.append(act)

            # Case b: object was added
            # TODO: do we care if added objects were flipped and changed size?
            elif obj.exists and not obj_before.exists:
                act = Action(name, 'added')
                actions.append(act)

            # Case c: object was edited (moved, flipped and/or resized)
            # TODO: some objects change x, y, z, flip without existing,
            # should investigate why
            elif obj.exists and obj_before.exists:
                if (obj.x != obj_before.x or obj.y != obj_before.y):
                    act = Action(name, 'moved')
                    actions.append(act)
                if obj.z != obj_before.z:
                    act = Action(name, 'resized')
                    actions.append(act)
                if obj.flip != obj_before.flip:
                    act = Action(name, 'flipped')
                    actions.append(act)
    if not actions and verbose:
        print('no changes')
    elif verbose:
        for act in actions:
            act.print_()
    return actions


class Game:
    """Represent a CoDraw game."""
    def __init__(self, game_id, game, crs, quick_load=False, with_peek=True):
        self.quick_load = quick_load
        self.with_peek = with_peek
        self.game_id = game_id
        self.id = int(game_id.split('_')[1])
        self.img_id = game['image_id']
        self.orig_scene = game['abs_t']
        self.n_turns = len(game['dialog'])
        if not quick_load:
            self.cliparts = get_objects(self.orig_scene)
        self.drawer_turns = []
        self.teller_turns = []
        self.scenes = []
        self.cr_turns = []
        self.peek_turn = None
        self.scores = []
        self.actions = []
        self.build_dialogue(game['dialog'], crs)
        lim = self.peek_turn or 10000
        self.cr_turns_before_peek = [t for t in self.cr_turns if t < lim]

    def build_dialogue(self, dialogue, crs):
        # add an empty scene to enable score computation
        # will be removed below
        self.scenes.append('')
        for i, turn in enumerate(dialogue):
            if 'peeked' in turn and not self.with_peek:
                self.n_turns = i
                break
            elif 'peeked' in turn:
                self.peek_turn = i
            self.teller_turns.append(turn['msg_t'])
            self.drawer_turns.append(turn['msg_d'])
            if turn['msg_d'] in crs:
                self.cr_turns.append(i)
            scene_string = turn['abs_d']
            if scene_string:
                # remove the useless comma in the end
                scene_string = turn['abs_d'][:-1]
            self.scenes.append(scene_string)
            if not self.quick_load:
                if turn['abs_d']:
                    pred = AbstractScene(turn['abs_d'])
                    target = AbstractScene(self.orig_scene)
                    score = scene_similarity(pred, target)
                    self.scores.append(score)
                else:
                    self.scores.append(None)
                turn_actions = drawer_actions(self.scenes[-2], self.scenes[-1])
                self.actions.append(turn_actions)
        # remove the first dummy empty scene
        self.scenes = self.scenes[1:]
        self.check()

    def check(self):
        assert len(self.drawer_turns) == self.n_turns
        assert len(self.teller_turns) == self.n_turns
        assert len(self.scenes) == self.n_turns
        # check that last scene is never an empty string
        assert len(self.scenes[-1]) > 50
        if not self.quick_load:
            assert len(self.scores) == self.n_turns
            assert self.scores[-1] is not None
            if self.peek_turn:
                assert self.scores[self.peek_turn - 1] is not None

    @property
    def n_crs(self):
        return len(self.cr_turns)

    @property
    def n_cliparts(self):
        return len(self.cliparts)

    @property
    def n_crs_before_peek(self):
        if not self.peek_turn:
            return len(self.cr_turns)
        return sum([turn < self.peek_turn for turn in self.cr_turns])

    @property
    def n_crs_after_peek(self):
        if not self.peek_turn:
            return 0
        return sum([turn >= self.peek_turn for turn in self.cr_turns])

    @property
    def n_actions(self):
        return sum([1 for turn in self.actions for act in turn])

    @property
    def n_actions_before_peek(self):
        stop = self.peek_turn or self.n_turns
        return sum([1 for turn in self.actions[:stop] for act in turn])

    def count_actions(self, only_before_peek=False, only_after_peek=False):
        action_counts = Counter()
        if not self.peek_turn and only_after_peek:
            return action_counts
        # an upper bound to avoid comparison to None 
        stop = self.peek_turn or 10000
        for t, turn in enumerate(self.actions):
            if t < stop and only_after_peek:
                continue
            if t == stop and only_before_peek:
                break
            if not turn:
                action_counts.update(['none'])
            for action in turn:
                action_counts.update([action.action])
        return action_counts

    def print_actions(self):
        for i in range(self.n_turns):
            cr = True if i in self.cr_turns else False
            if self.peek_turn == i:
                cr = f'{cr}  PEEK'
            print(f'\n### {i}, CR={cr}')
            for act in self.actions[i]:
                act.print_()

    def get_edits_track(self):
        edits = {}
        for t, turn in enumerate(self.actions):
            for action in turn:
                clip = action.clipart
                if clip not in edits:
                    assert action.action == 'added'
                    edits[clip] = EditTrack(t, [], [])
                else:
                    assert (action.action != 'added'
                            or 'deleted' in edits[clip].edits)
                    edits[clip].edit_turns.append(t)
                    edits[clip].edits.append(action.action)
        return edits

    def get_last_edits(self):
        seq = self.get_edits_track()
        return {clipart: track.edit_turns[-1] - track.added_at
                if track.edits else None for clipart, track in seq.items()}

    def get_dialogue(self):
        return [Turn(self.teller_turns[t], self.drawer_turns[t])
                for t in range(self.n_turns)]

    def get_dialogue_string(self, sep_t='/T', sep_d='/D'):
        dialogue = self.get_dialogue()
        context = " ".join([f'{sep_t} {turn.teller} {sep_d} {turn.drawer} '
                           for turn in dialogue])
        return context

    def print_dialogue(self):
        for i in range(self.n_turns):
            if self.peek_turn == i:
                print('\n --- PEEKED --- \n')
            print(f'TELLER: {self.teller_turns[i]}')
            print(f'DRAWER: {self.drawer_turns[i]}\n')

    @property
    def score_diffs(self):
        # replace None by 0
        scores = [s or 0 for s in self.scores]
        return np.array(scores) - np.array([0] + scores[:-1])

    def plot_dialogue(self):
        x = [-1] + list(range(self.n_turns))
        # replace None values by 0
        y = [0] + [s or 0 for s in self.scores]
        plt.plot(x, y, linestyle='-')
        plt.xticks(x)
        plt.yticks(np.arange(0, 6, 0.5))
        plt.ylim((0, 5))

        cr_scores = [self.scores[i] for i in self.cr_turns]
        plt.plot(self.cr_turns, cr_scores, 'ro', label='clarification')

        other_turns = [i for i in range(self.n_turns) if i not in self.cr_turns]
        other_scores = [self.scores[i] for i in other_turns]
        plt.plot(other_turns, other_scores, 'bo', label='')

        if self.peek_turn:
            # TODO: deal with cases when the last score is None
            plt.axvline(self.peek_turn - 0.5, color='g', 
                        linestyle='--', label='peeked')

        plt.xlabel('turn')
        plt.ylabel('scene similarity score')
        plt.title(self.game_id)
        plt.legend()
        plt.show()
