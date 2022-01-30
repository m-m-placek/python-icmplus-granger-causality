# Michal M. Placek, last modified 2021-12-20, version 1.0

import unittest
from gct_MP import var_specrad, acf_to_var, acf_to_gc, find_longest_streak, find_longest_gap, invalid_data_percentage, \
    calculate_gc_expF_from_input, fill_missing_multichannel_data
from statsmodels.tsa.vector_ar.var_model import var_acf
from statsmodels.tsa.api import VAR
import numpy as np
import math


class TestVarSpecRad(unittest.TestCase):
    def test_var_specrad_ord2_stable(self):
        A = [[[ 0.028330194776083,  0.001534395859038],
              [ 0.035388665426296, -0.0203560115817  ]],
             [[-0.019614550833825,  0.571779546844646],
              [-0.205213381643853,  0.168304580637164]]]
        expected_rho = 0.5868523169090906
        obtained_rho = var_specrad(A)
        self.assertAlmostEqual(obtained_rho, expected_rho, places=13)

    def test_var_specrad_ord3_stable(self):
        A = [[[ 0.085308411763804,  0.004206757978148],
              [ 0.112588950784858,  0.005737218125848]],
             [[-0.014436003632184,  0.560230761964757],
              [-0.211629088856529,  0.169330715515973]],
             [[ 0.021980603149471, -0.037313622541474],
              [ 0.163325484815279, -0.096098674536687]]]
        expected_rho = 0.6895637231693339
        obtained_rho = var_specrad(A)
        self.assertAlmostEqual(obtained_rho, expected_rho, places=13)

    def test_var_specrad_ord15_slightlyUnstable(self):
        A = [[[ 9.955300285505437e-1,  6.351208518171459e-1],
              [ 5.009236716928991e-1,  9.278004830290840e-1]],
             [[-1.348658670888221e+0, -6.049366352154507e-1],
              [-8.823607942279480e-1, -1.177725599414283e+0]],
             [[ 1.876075828117845e+0,  1.109789709532901e+0],
              [ 6.827991453433858e-1,  2.002142589120297e+0]],
             [[-1.703936879427884e+0, -1.227152529020998e+0],
              [-8.989385787955185e-1, -2.003076384615080e+0]],
             [[ 2.332763506609298e+0,  1.113172008259423e+0],
              [ 1.142839173780861e+0,  2.089840513118643e+0]],
             [[-2.333202088933541e+0, -1.109928205622808e+0],
              [-1.414275038170599e+0, -1.852673113852039e+0]],
             [[ 2.702116969978322e+0,  7.311414484964341e-1],
              [ 1.629672945784242e+0,  1.693361610411239e+0]],
             [[-2.368480289804821e+0, -7.486778964153095e-1],
              [-1.557474350501315e+0, -1.321668111219085e+0]],
             [[ 2.072807070280431e+0,  4.582789015367462e-1],
              [ 1.268406843914553e+0,  1.247386988999909e+0]],
             [[-1.708923949785484e+0, -6.843966011426440e-1],
              [-1.038959348845141e+0, -1.117604111795012e+0]],
             [[ 1.530789431983513e+0,  7.221352881040362e-2],
              [ 1.196492161178431e+0,  4.444602240569948e-1]],
             [[-1.367341737447771e+0,  1.435090768055247e-1],
              [-9.306082406462758e-1, -1.342046992480897e-1]],
             [[ 3.745257693259731e-1,  2.364658802283132e-1],
              [ 2.554739365976237e-1,  3.823291464680985e-1]],
             [[-4.080794699290571e-1, -7.159768085837348e-4],
              [-1.784269395425771e-1, -1.782836643018725e-1]],
             [[ 3.172833288588131e-1, -1.267844015750621e-1],
              [ 1.877667634646650e-1, -4.924163176099755e-3]]]
        expected_rho = 1.0000094250389457
        obtained_rho = var_specrad(A)
        self.assertAlmostEqual(obtained_rho, expected_rho, places=11)

    def test_var_specrad_ord14_yetStable(self):
        A = [[[ 1.133652362444308e+0,  3.640588681391582e-1],
              [ 6.154239127193212e-1,  6.871875425681946e-1]],
             [[-1.091754459129008e+0, -6.406981931532909e-1],
              [-6.962122970454936e-1, -1.149948052604001e+0]],
             [[ 1.644584549289468e+0,  8.391278701004412e-1],
              [ 4.480310490563929e-1,  1.778978317106739e+0]],
             [[-1.558358731582442e+0, -7.434245763880768e-1],
              [-8.228012109185180e-1, -1.462491861601193e+0]],
             [[ 1.973884415681586e+0,  4.491583279774456e-1],
              [ 9.049539907353804e-1,  1.396370955239501e+0]],
             [[-1.910360550962411e+0, -5.748325122392843e-1],
              [-1.115728233567842e+0, -1.224219875580441e+0]],
             [[ 2.063858683194099e+0,  3.306776859193849e-1],
              [ 1.205932906452958e+0,  1.169861886452545e+0]],
             [[-2.100713529917210e+0,  3.545055787850154e-2],
              [-1.441538455789541e+0, -4.342732664776452e-1]],
             [[ 1.327606075533927e+0,  3.425194343458316e-1],
              [ 8.078338489208607e-1,  9.086088761086427e-1]],
             [[-1.108909256077017e+0, -3.217592318523328e-1],
              [-6.229389806344060e-1, -6.398767019227773e-1]],
             [[ 1.121542462287656e+0, -2.145236427437782e-1],
              [ 1.082148788747280e+0, -1.156622844473395e-1]],
             [[-6.806384561708521e-1,  1.528622390506145e-1],
              [-4.614889402253811e-1,  6.195958217292560e-3]],
             [[ 2.782352522531916e-1, -9.923009241184298e-3],
              [ 2.266303933039227e-1,  4.956471356398766e-2]],
             [[-1.454589063632296e-1,  9.572224894251607e-4],
              [-1.815832989866586e-1,  3.870714026540527e-2]]]
        expected_rho = 0.9999192369881642
        obtained_rho = var_specrad(A)
        self.assertAlmostEqual(obtained_rho, expected_rho, places=11)

    def test_var_specrad_5variables_ord3_stable(self):
        # A = [[[ 1.34978305e+0, -3.60227135e-2,  1.32499524e-2, -3.45193892e-2,  3.22229433e-2],
        #       [-1.90853319e-2,  6.20477283e-2, -1.50798159e-2,  1.92347462e-2,  2.10504991e-2],
        #       [ 6.27699125e-2,  4.07598144e-2, -2.48640420e-2, -4.38769188e-2,  1.97819513e-2],
        #       [ 5.12786930e-3, -4.45963194e-3,  1.38447691e-2,  3.15655157e-1,  3.78818980e-1],
        #       [-2.19834837e-2, -4.38457722e-3,  1.40226791e-2, -3.70969419e-1,  3.45093437e-1]],
        #      [[-9.39727873e-1, -7.01402984e-3, -8.14561578e-3, -4.34552160e-3,  5.89934190e-2],
        #       [ 5.20344010e-1, -1.05375046e-2,  8.36408095e-3,  3.13160783e-2, -3.03765149e-3],
        #       [-8.76636944e-2, -3.19089382e-2,  5.03935084e-3,  3.58465630e-2,  7.76089220e-2],
        #       [-5.40016679e-1,  2.65675398e-2,  1.59730663e-2,  3.57250669e-2, -4.19481940e-2],
        #       [ 3.06390814e-2,  6.08174766e-2,  2.71380436e-4, -4.41541463e-2, -6.68059621e-3]],
        #      [[ 1.93909853e-2,  1.14358333e-3,  1.64935398e-2,  4.00780837e-2,  1.15167914e-4],
        #       [-3.28342703e-2,  3.40404875e-2,  2.31608991e-2, -1.37336862e-2, -7.42864005e-3],
        #       [-3.94110102e-1, -1.03441320e-2,  1.37609932e-2, -1.95356770e-2, -5.32868657e-2],
        #       [ 3.89499404e-2, -7.39367153e-2, -2.68890399e-2, -3.83371461e-2,  4.78717582e-2],
        #       [-5.04298580e-2, -1.71589603e-2,  7.86556639e-3,  4.17837141e-2,  5.10264060e-2]]]
        # A = np.zeros((3,5,5))
        # A[0,0,0] = 1.34350288425444
        # A[0,3,3] = A[0,3,4] = A[0,4,4] = 0.353553390593274
        # A[0,4,3] = -A[0,3,3]
        # A[1,0,0] = -0.9025
        # A[1,1,0] = 0.5
        # A[1,3,0] = -0.5
        # A[2,2,0] = -0.4
        # expected_rho = 0.9498931790669959
        A = [[[1.34350288,  0,        0,           0,           0],
              [0,           0,        0,           0,           0],
              [0,           0,        0,           0,           0],
              [0,           0,        0,  0.35355339,  0.35355339],
              [0,           0,        0, -0.35355339,  0.35355339]],
             [[-0.9025,     0,        0,           0,           0],
              [0.5,         0,        0,           0,           0],
              [0,           0,        0,           0,           0],
              [-0.5,        0,        0,           0,           0],
              [0,           0,        0,           0,           0]],
             [[0,           0,        0,           0,           0],
              [0,           0,        0,           0,           0],
              [-0.4,        0,        0,           0,           0],
              [0,           0,        0,           0,           0],
              [0,           0,        0,           0,           0]]]
        expected_rho = 0.95
        obtained_rho = var_specrad(A)
        self.assertAlmostEqual(obtained_rho, expected_rho, places=13)

    def test_var_to_acf_and_back(self):
        A = [[[0.08530841, 0.00420676],
              [0.11258895, 0.00573722]],
             [[-0.014436, 0.56023076],
              [-0.21162909, 0.16933072]],
             [[0.0219806, -0.03731362],
              [0.16332548, -0.09609867]]]
        SIG = [[ 0.56869792, -0.01690957],
               [-0.01690957, 2.12883914]]
        ac_lags = 50
        G = var_acf(np.array(A), SIG, nlags=ac_lags)
        A_reconstr, SIG_reconstr = acf_to_var(G)
        self.assertTrue( np.allclose(SIG, SIG_reconstr) )
        self.assertTrue( np.allclose(A, A_reconstr[:3, :, :]) )

    def test_var_to_acf_to_gc(self):
        A = [[[0.08530841, 0.00420676],
              [0.11258895, 0.00573722]],
             [[-0.014436, 0.56023076],
              [-0.21162909, 0.16933072]],
             [[0.0219806, -0.03731362],
              [0.16332548, -0.09609867]]]
        SIG = [[ 0.56869792, -0.01690957],
               [-0.01690957, 2.12883914]]
        ac_lags = 50
        G = var_acf(np.array(A), SIG, nlags=ac_lags)
        expected_expF = np.array([[np.nan, 2.20060127],
                                  [1.01974069, np.nan]])
        obtained_expF = acf_to_gc(G)
        self.assertTrue( np.allclose(expected_expF, obtained_expF, equal_nan=True) )

    # this test checks all the path from input data to F-mag calculated using calculate_gc_expF_from_input() function
    def test_input_to_Fmag(self):
        X = np.array([[0.4145818, -1.44515969],
               [1.71079968, -0.05254712],
               [-2.3819322, 0.39216972],
               [0.73908798, -1.87890558],
               [0.1956799, -0.15675959],
               [-1.43077363, -0.28439045],
               [-0.55667736, 4.0416739],
               [0.21953913, 3.4821118],
               [3.4553116, 0.33531249],
               [2.64635169, 2.81411157],
               [-1.47297228, -1.4396394],
               [2.91183813, -0.92933198],
               [0.60231889, 2.04264887],
               [-0.18614021, -1.3038097],
               [0.59165757, 0.81012116],
               [-0.3280514, 1.58705905],
               [-0.24722969, 2.81904275],
               [1.36661227, -0.57039594],
               [1.28594915, 0.44711055],
               [1.29410708, -2.44202077],
               [0.5484118, 3.59855832],
               [-1.33057226, 2.42876583],
               [0.59415331, 1.84117739],
               [1.50714995, -0.05017561],
               [0.36580843, 0.23158119],
               [0.91160767, -0.60259869],
               [0.6037998, 1.36560773],
               [-0.42652626, -1.09184309],
               [0.17078613, 1.56324873],
               [-0.91036814, -3.22557476],
               [0.76531029, -1.44940881],
               [-1.27015544, -1.65977358],
               [-1.1919558, -4.54802954],
               [-0.93258403, 1.91966659],
               [-3.0673695, 0.58048625],
               [1.31529496, -0.74813679],
               [0.2021052, 0.00993224],
               [-0.87801366, -0.6107125],
               [1.2472132, 0.22124861],
               [-1.83460176, -0.56720143],
               [-0.22532778, 0.31540817],
               [-0.36453238, 0.0241748],
               [0.1961214, -2.64178064],
               [0.18977326, -0.34239063],
               [-0.98796525, -1.02293389],
               [-0.15313663, -0.37818738],
               [-0.28796436, -0.08982434],
               [0.50462195, 0.54902783],
               [0.97018033, -2.89297692],
               [0.98618796, 1.01490016],
               [-0.98673816, -0.7207453],
               [-0.04572625, -1.16021695],
               [-1.33720238, -0.06830877],
               [-1.23658608, 0.70777837],
               [-0.12993467, 0.22233101],
               [1.40954497, 0.21147298],
               [-0.89275125, -0.96680292],
               [0.24829348, 2.44205355],
               [-0.34866974, -1.34052371],
               [0.9942708, -0.58315992],
               [-1.21214963, 0.23208507],
               [-0.09052787, 0.22599562],
               [0.42944168, 0.39739524],
               [0.97752488, 2.58524247],
               [1.42112656, 0.13721892],
               [-0.0371542, -0.46145504],
               [-1.61467565, -2.34538845],
               [-0.86538717, 1.45830084],
               [-1.18466707, -0.81882435],
               [2.22737189, 1.51280449],
               [-0.73868722, -1.55111129],
               [0.62499145, -1.46794509],
               [-0.31550385, -2.24063489],
               [0.76552509, -1.09544649],
               [-0.88793457, -1.05771068],
               [-1.52535431, 0.91318497],
               [-1.54546126, -0.33434817],
               [0.36510857, -0.03905275],
               [-0.30046049, 0.91646878],
               [-0.31913883, 1.12691763],
               [1.29622481, -1.19057178],
               [0.16849904, 2.01604029],
               [0.07472572, -1.67986369],
               [1.46461375, 0.33504912],
               [-0.92755129, 1.36118315],
               [0.57353908, 0.76868716],
               [0.71200283, -0.92865695],
               [-0.36680048, 0.06812642],
               [0.09258475, -0.514498],
               [-1.28892927, 0.0103208],
               [-1.27103812, 0.50034448],
               [-0.01821062, 2.34119654],
               [0.59916869, -0.99668665],
               [2.46240592, 0.18368829],
               [-0.78997601, -0.05789243],
               [0.06424569, -1.13364849],
               [-0.20557976, 1.06132218],
               [-2.05610826, -1.35445276],
               [-0.56205149, 0.2635983],
               [-1.91776418, 0.098502]])
        # mdl = VAR(X)
        # var_fit_result = mdl.fit(maxlags=2, trend='nc', method='ols')
        # rho = var_specrad(var_fit_result.coefs)
        # ac_dec_tol = 1E-8
        # ac_lags_limit = 10000
        # ac_dec_lags = math.ceil(math.log(ac_dec_tol) / math.log(rho))
        # ac_lags = min(ac_dec_lags, ac_lags_limit)
        # G = var_acf(var_fit_result.coefs, var_fit_result.sigma_u_mle, nlags=ac_lags)
        # expF = acf_to_gc(G)
        expF = calculate_gc_expF_from_input(X, order=2, ac_dec_tol=1E-8, ac_lags_limit=10000)
        obtained_F_mag = np.log(expF)
        expected_F_mag = np.array([[    np.nan, 0.79186887],
                                   [0.01175751,     np.nan]])
        self.assertTrue( np.allclose(expected_F_mag, obtained_F_mag, equal_nan=True) )
        # F_mag is the same as in Matlab
        # [A, SIG, E] = tsdata_to_var(X.', 2, 'OLS');
        # [G, info] = var_to_autocov(A, SIG, 10000);
        # F_mag = autocov_to_pwcgc(G);


class TestUtils(unittest.TestCase):
    def test_find_longest_streak_0(self):
        seq = [False] * 10
        expected = 0
        obtained = find_longest_streak(seq)
        self.assertEqual(obtained, expected)

    def test_find_longest_streak_1(self):
        seq = [1,0,1,0,1,0,0,1]
        expected = 1
        obtained = find_longest_streak(seq)
        self.assertEqual(obtained, expected)

    def test_find_longest_streak_3(self):
        seq = np.array([0,0,0,0,1,1,1,0,1,1,0,0,0], dtype=bool)
        expected = 3
        obtained = find_longest_streak(seq)
        self.assertEqual(obtained, expected)

    def test_find_longest_streak_allTrue20(self):
        seq = [True] * 20
        expected = 20
        obtained = find_longest_streak(seq)
        self.assertEqual(obtained, expected)

    def test_find_longest_gap_4(self):
        n = np.NaN
        sig = [3.4, 5.6, n,n,n, -2.3, n,n,n, np.Inf, 0, -1, n,n]
        expected = 4
        obtained = find_longest_gap(sig)
        self.assertEqual(obtained, expected)

    def test_invalid_data_percentage_20(self):
        n = np.NaN
        sig = [1,2,3, np.NaN, 4,5, np.NaN, np.Inf, 6,7,8,9,10,11,12,13,14,15,16, np.NaN]
        expected = 20
        obtained = invalid_data_percentage(sig)
        self.assertEqual(obtained, expected)

    def test_fill_missing_multichannel_data_linear(self):
        n = np.nan
        sig1 = [n, n, n, n, 9, n, 8, n, 6, 4, 5, 6, n, 6, 1, n, 0, 0, 0]
        sig2 = [0, 0, 0, 0, 6, 7, 8, n, 3, 2, 1, 0, n, 6, 2, n, n, n, np.inf]
        expected = np.array([[9. , 6. ],
                             [9. , 6. ],
                             [9. , 6. ],
                             [9. , 6. ],
                             [9. , 6. ],
                             [8.5, 7. ],
                             [8. , 8. ],
                             [7. , 5.5],
                             [6. , 3. ],
                             [4. , 2. ],
                             [5. , 1. ],
                             [6. , 0. ],
                             [6. , 3. ],
                             [6. , 6. ],
                             [1. , 2. ],
                             [1. , 2. ],
                             [1. , 2. ],
                             [1. , 2. ],
                             [1. , 2. ]])
        obtained = fill_missing_multichannel_data((sig1, sig2), fill_method='linear', alter_all_channels=True)
        self.assertTrue(np.all(obtained == expected))


if __name__ == '__main__':
    unittest.main()
