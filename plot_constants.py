COLORS = [
    (.4, .4, .4),

    (1., 0., 0.),
    (0., 0., 1.),
    (0., 1., 0.),

    (.6, 0., .6),
    (0., .6, .6),
    (.6, .6, 0.),

    (1., 0., .5),
    (.5, 0., 1.),
    (0., 1., .5),
    (0., .5, 1.),
    (1., .5, 0.),
    (.5, 1., 0.),
]

C_OVER_TWENTY_RULE = (0.038,0.062)
C_OVER_TWO_RULE = (0.38,0.62)
C_RULE = (0.8,1.4)
TWO_C_RULE = (1.6,2.4)
THREE_C_RULE = (2.4,3.6)


#TODO(sam): keep track of these in the database and allow users to modify.
Preferred_Legends = {
    (C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, None, None):0,
    (C_OVER_TWENTY_RULE, C_OVER_TWO_RULE, C_OVER_TWO_RULE, None, None): 1,
    (C_OVER_TWENTY_RULE, C_RULE, C_RULE, None, None): 2,
    (C_OVER_TWENTY_RULE, TWO_C_RULE, TWO_C_RULE, None, None): 3,
    (C_OVER_TWENTY_RULE, THREE_C_RULE, THREE_C_RULE, None, None): 4,

    (C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, C_OVER_TWENTY_RULE, None, None): 0,
    (C_OVER_TWENTY_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 5,
    (C_OVER_TWO_RULE, C_RULE, None, None, None): 1,
    (C_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 2,
    (TWO_C_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 3,
    (THREE_C_RULE, C_RULE, C_OVER_TWENTY_RULE, None, None): 4,

}
