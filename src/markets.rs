//! Credit, labour and goods markets, BeforeIT `src/markets/` (appendix
//! A.4.1, A.2.1 and A.1.1).

use crate::julia::{max, min, pos, sum, sum_by, zero_to_one};
use crate::random::{Randomness, WeightVector};
use crate::{AbstractModel, Extension, Firms, Government, Properties, RestOfTheWorld};

/// Weight of each seller when a buyer picks one, `create_weighted_sampler`:
/// cheaper and larger sellers are picked more often.
fn seller_weights(P_f: &[f64], S_f: &[f64], F_g: &[usize]) -> WeightVector {
    if F_g.is_empty() {
        return WeightVector::new(Vec::new());
    }
    let price: Vec<f64> = F_g.iter().map(|&f| libm::exp(-2.0 * P_f[f])).collect();
    let price_sum = sum(&price);
    let size_sum = sum_by(F_g.len(), |k| S_f[F_g[k]]);
    let w = F_g
        .iter()
        .zip(&price)
        .map(|(&f, &p)| pos(p / price_sum) + S_f[f] / size_sum)
        .collect();
    WeightVector::new(w)
}

/// All sellers on the goods markets, `initialize_variables_firms_market`:
/// first the firms, then one foreign seller `I + g` for each product `g`.
struct Sellers {
    /// Price.
    P_f: Vec<f64>,
    /// What the seller offers: production plus inventories, for a foreign
    /// seller the supply of imports.
    S_f: Vec<f64>,
    /// Spare capacity `K_i·κ_i − Y_i` that records demand which cannot be
    /// served; unlimited for foreign sellers.
    S_f_: Vec<f64>,
    /// Sellers of each product, in ascending order.
    by_good: Vec<Vec<usize>>,
}

impl Sellers {
    fn new<X>(m: &AbstractModel<X>) -> Sellers {
        let (f, rotw, G) = (&m.firms, &m.rotw, m.prop.G);
        let I = f.len();
        let P_f = f.P_i.iter().chain(&rotw.P_m).copied().collect();
        let S_f = (0..I)
            .map(|i| f.Y_i[i] + f.S_i[i])
            .chain(rotw.Y_m.iter().copied())
            .collect();
        let S_f_ = (0..I)
            .map(|i| f.K_i[i] * f.kappa_i[i] - f.Y_i[i])
            .chain(std::iter::repeat_n(f64::INFINITY, G))
            .collect();
        let mut by_good = vec![Vec::new(); G];
        for i in 0..I {
            by_good[f.G_i[i]].push(i);
        }
        for (g, sellers) in by_good.iter_mut().enumerate() {
            sellers.push(I + g);
        }
        Sellers {
            P_f,
            S_f,
            S_f_,
            by_good,
        }
    }

    /// The sellers of product g in the second round, `F_g_`: those with spare
    /// capacity left and a positive supply.
    fn second_round(&self, g: usize, S_fg_: &[f64]) -> Vec<usize> {
        self.by_good[g]
            .iter()
            .copied()
            .filter(|&f| S_fg_[f] > 0.0 && self.S_f[f] > 0.0)
            .collect()
    }
}

/// State of one product during its market, shared by the firms market and
/// the retail market (`S_fg`, `S_fg_` and `F_g` in `perform_market!`). Only
/// the entries of the product's sellers are read and written.
struct GoodState {
    /// What each seller has left.
    S_fg: Vec<f64>,
    /// Spare capacity left with each seller.
    S_fg_: Vec<f64>,
    /// Sellers that still have something to sell.
    F_g: Vec<usize>,
}

impl GoodState {
    fn new(s: &Sellers) -> GoodState {
        GoodState {
            S_fg: s.S_f.clone(),
            S_fg_: s.S_f_.clone(),
            F_g: Vec::new(),
        }
    }

    /// Fresh state for product g. Julia copies all of `S_f` and `S_f_`;
    /// here only the entries of the product's sellers are reset, which gives
    /// the same values for them.
    fn reset(&mut self, s: &Sellers, g: usize) {
        self.F_g.clear();
        for &f in &s.by_good[g] {
            self.S_fg[f] = s.S_f[f];
            self.S_fg_[f] = s.S_f_[f];
            self.F_g.push(f);
        }
    }
}

/// Results of the goods markets, summed over all products as in
/// `update_aggregate_variables!`. The sums run over the products in their
/// order, as Julia's `sum(·, dims = 2)` does.
struct MarketTotals {
    I_i: Vec<f64>,
    DM_i: Vec<f64>,
    P_bar_i: Vec<f64>,
    P_CF_i: Vec<f64>,
    Q_d_i: Vec<f64>,
    Q_d_m: Vec<f64>,
    C_h: Vec<f64>,
    I_h: Vec<f64>,
    C_j: f64,
    C_l: f64,
    P_bar_h: f64,
    P_bar_CF_h: f64,
    P_j: f64,
    P_l: f64,
}

/// The results of the market of one product: column `g` of Julia's
/// matrices `I_i_g`, `DM_i_g`, `P_bar_i_g`, `P_CF_i_g`, `Q_d_i_g` and
/// `Q_d_m_g`, entry `g` of `C_j_g` and so on, and the product's share `b`
/// and `d` of the households' consumption and investment.
struct GoodTotals {
    I_i: Vec<f64>,
    DM_i: Vec<f64>,
    P_bar_i: Vec<f64>,
    P_CF_i: Vec<f64>,
    C_h: Vec<f64>,
    I_h: Vec<f64>,
    /// Demand met by each seller of the product.
    Q_d: Vec<(usize, f64)>,
    C_j: f64,
    C_l: f64,
    P_bar_h: f64,
    P_bar_CF_h: f64,
    P_j: f64,
    P_l: f64,
}

impl GoodTotals {
    fn new(I: usize, H: usize) -> GoodTotals {
        GoodTotals {
            I_i: vec![0.0; I],
            DM_i: vec![0.0; I],
            P_bar_i: vec![0.0; I],
            P_CF_i: vec![0.0; I],
            C_h: vec![0.0; H],
            I_h: vec![0.0; H],
            Q_d: Vec::new(),
            C_j: 0.0,
            C_l: 0.0,
            P_bar_h: 0.0,
            P_bar_CF_h: 0.0,
            P_j: 0.0,
            P_l: 0.0,
        }
    }
}

impl MarketTotals {
    fn new(I: usize, G: usize, H: usize) -> MarketTotals {
        MarketTotals {
            I_i: vec![0.0; I],
            DM_i: vec![0.0; I],
            P_bar_i: vec![0.0; I],
            P_CF_i: vec![0.0; I],
            Q_d_i: vec![0.0; I],
            Q_d_m: vec![0.0; G],
            C_h: vec![0.0; H],
            I_h: vec![0.0; H],
            C_j: 0.0,
            C_l: 0.0,
            P_bar_h: 0.0,
            P_bar_CF_h: 0.0,
            P_j: 0.0,
            P_l: 0.0,
        }
    }

    /// Adds the results of one product. The products are added in their
    /// order, so the totals do not depend on whether the markets ran in
    /// parallel.
    fn add(&mut self, r: &GoodTotals) {
        let I = self.I_i.len();
        for i in 0..I {
            self.DM_i[i] += r.DM_i[i];
            self.I_i[i] += r.I_i[i];
            self.P_bar_i[i] += r.P_bar_i[i];
            self.P_CF_i[i] += r.P_CF_i[i];
        }
        for h in 0..self.C_h.len() {
            self.C_h[h] += r.C_h[h];
            self.I_h[h] += r.I_h[h];
        }
        for &(f, q) in &r.Q_d {
            if f < I {
                self.Q_d_i[f] += q;
            } else {
                self.Q_d_m[f - I] += q;
            }
        }
        // These totals add up over the products from left to right, which is
        // Julia's `sum` over fewer than 1024 products.
        self.C_j += r.C_j;
        self.C_l += r.C_l;
        self.P_bar_h += r.P_bar_h;
        self.P_bar_CF_h += r.P_bar_CF_h;
        self.P_j += r.P_j;
        self.P_l += r.P_l;
    }
}

/// Shares of the products in a budget, weighted with last quarter's prices,
/// `initialize_variables_retail_market`.
fn price_weighted_shares(P_bar_g: &[f64], share: &[f64]) -> Vec<f64> {
    let weighted: Vec<f64> = P_bar_g.iter().zip(share).map(|(p, s)| p * s).collect();
    let total = sum(&weighted);
    weighted.iter().map(|x| x / total).collect()
}

/// Budgets for the retail market, `initialize_variables_retail_market`: all
/// households in the order workers, inactive persons, firm owners, bank
/// owners, and the shares of the products in the budgets for consumption,
/// housing investment, government and exports.
struct RetailDemand {
    C_d_h: Vec<f64>,
    I_d_h: Vec<f64>,
    b_HH_g: Vec<f64>,
    b_CFH_g: Vec<f64>,
    c_E_g: Vec<f64>,
    c_G_g: Vec<f64>,
}

impl RetailDemand {
    fn new<X>(m: &AbstractModel<X>) -> RetailDemand {
        let (w_act, w_inact, f, b) = (&m.w_act, &m.w_inact, &m.firms, &m.banks);
        let C_d_h = w_act
            .C_d_h
            .iter()
            .chain(&w_inact.C_d_h)
            .chain(&f.C_d_h)
            .chain(&b.C_d_h)
            .copied()
            .collect();
        let I_d_h = w_act
            .I_d_h
            .iter()
            .chain(&w_inact.I_d_h)
            .chain(&f.I_d_h)
            .chain(&b.I_d_h)
            .copied()
            .collect();
        let (P_bar_g, p) = (&m.agg.P_bar_g, &m.prop);
        RetailDemand {
            C_d_h,
            I_d_h,
            b_HH_g: price_weighted_shares(P_bar_g, &p.b_HH_g),
            b_CFH_g: price_weighted_shares(P_bar_g, &p.b_CFH_g),
            c_E_g: price_weighted_shares(P_bar_g, &p.c_E_g),
            c_G_g: price_weighted_shares(P_bar_g, &p.c_G_g),
        }
    }
}

/// `search_and_matching_credit!` (A.62 to A.64): bank by bank, the firms of
/// the bank that want new loans are served in random order, limited by the
/// loan-to-value ratio and by the bank's equity relative to its exposure
/// ([`crate::base::bank_exposure`] with expected loans). As in the Julia
/// code, the new loans are stored at the position given by the firm's ID.
pub fn search_and_matching_credit<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
) {
    let (zeta, zeta_LTV) = (m.prop.zeta, m.prop.zeta_LTV);
    let mut DL_i = vec![0.0; m.firms.DL_i.len()];
    for bank_id in 0..m.prop.n_banks {
        let f = &m.firms;
        let bank_firms: Vec<usize> = m.banks.firms[bank_id]
            .iter()
            .map(|&id| f.index(id))
            .filter(|&i| f.DL_d_i[i] > 0.0)
            .collect();
        let DL_d_i: Vec<f64> = bank_firms.iter().map(|&i| f.DL_d_i[i]).collect();
        let K_e_i: Vec<f64> = bank_firms.iter().map(|&i| f.K_e_i[i]).collect();
        let L_e_i: Vec<f64> = bank_firms.iter().map(|&i| f.L_e_i[i]).collect();
        let f_id: Vec<usize> = bank_firms.iter().map(|&i| f.ID[i]).collect();
        let E_k = m.banks.E_k[bank_id];

        // The capital constraint binds on the bank's whole loan book, not
        // only on the firms that ask for new loans.
        let exposure = X::bank_exposure(m, bank_id, true, m.prop.overdrafts_in_capital_ratio);

        let mut s_DL = 0.0;
        let mut I_FG: Vec<usize> = (0..DL_d_i.len()).collect();
        rng.shuffle(&mut I_FG);
        for i in I_FG {
            DL_i[f_id[i]] = max(
                0.0,
                min(
                    min(DL_d_i[i], zeta_LTV * K_e_i[i] - L_e_i[i]),
                    E_k / zeta - exposure - s_DL,
                ),
            );
            s_DL += DL_i[f_id[i]];
        }
    }
    m.firms.DL_i = DL_i;
}

/// `search_and_matching_labour!` (A.23, A.24): firms with too many employees
/// dismiss randomly chosen ones; then firms with vacancies, in random order,
/// hire one unemployed person each per round until the vacancies or the
/// unemployed run out.
pub fn search_and_matching_labour<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
) {
    {
        let f = &mut m.firms;
        let O_h = &mut m.w_act.O_h;
        let mut V_i: Vec<i64> = (0..f.len()).map(|i| f.N_d_i[i] - f.N_i[i]).collect();

        let mut H_E: Vec<usize> = (0..O_h.len()).filter(|&h| O_h[h].is_some()).collect();
        rng.shuffle(&mut H_E);
        for h in H_E {
            if let Some(i) = O_h[h]
                && V_i[i] < 0
            {
                O_h[h] = None;
                f.N_i[i] -= 1;
                V_i[i] += 1;
            }
        }

        let mut H_U: Vec<usize> = (0..O_h.len()).filter(|&h| O_h[h].is_none()).collect();
        rng.shuffle(&mut H_U);
        let mut I_V: Vec<usize> = (0..f.len()).filter(|&i| V_i[i] > 0).collect();
        let mut next = 0;
        while next < H_U.len() && !I_V.is_empty() {
            rng.shuffle(&mut I_V);
            for &i in &I_V {
                O_h[H_U[next]] = Some(i);
                f.N_i[i] += 1;
                V_i[i] -= 1;
                next += 1;
                if next == H_U.len() {
                    break;
                }
            }
            I_V.retain(|&i| V_i[i] > 0);
        }
    }
}

/// `search_and_matching!` (appendix A.1.1, and A.1, A.2, A.15, A.16, A.19,
/// A.20, A.47, A.48, A.50, A.51, A.57, A.58, A.79, A.80, A.83, A.84): for
/// each product, first the firms market for intermediate inputs and
/// investment goods, then the retail market for households, government and
/// foreign consumers. Buyers come in random order and pick sellers by price
/// and size. With `parallel`, the products' markets run in parallel threads
/// (see [`Randomness::split`]).
pub fn search_and_matching<X: Extension, R: Randomness + ?Sized>(
    m: &mut AbstractModel<X>,
    rng: &mut R,
    parallel: bool,
) {
    m.goods_markets(rng, parallel);
}

impl<X: Extension> AbstractModel<X> {
    /// `search_and_matching_credit!(model)`; see
    /// [`crate::base::search_and_matching_credit`].
    pub fn search_and_matching_credit<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        X::search_and_matching_credit(self, rng)
    }

    /// `search_and_matching_labour!(model)`; see
    /// [`crate::base::search_and_matching_labour`].
    pub fn search_and_matching_labour<R: Randomness + ?Sized>(&mut self, rng: &mut R) {
        X::search_and_matching_labour(self, rng)
    }

    /// `search_and_matching!(model)`; see
    /// [`crate::base::search_and_matching`].
    pub fn search_and_matching<R: Randomness + ?Sized>(&mut self, rng: &mut R, parallel: bool) {
        X::search_and_matching(self, rng, parallel)
    }

    /// The body of `search_and_matching!`. With `parallel`, the markets of
    /// the products run in parallel threads, each with its own generator
    /// split off from `rng` in the order of the products (Julia's
    /// `@maybe_threads`, where each task has its own generator).
    fn goods_markets<R: Randomness + ?Sized>(&mut self, rng: &mut R, parallel: bool) {
        let sellers = Sellers::new(self);
        let demand = RetailDemand::new(self);
        let (G, I, H) = (self.prop.G, self.firms.len(), demand.C_d_h.len());
        let mut t = MarketTotals::new(I, G, H);
        if parallel {
            let mut rngs: Vec<Box<dyn Randomness + Send>> = (0..G).map(|_| rng.split()).collect();
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let this = &self.market_view();
            let (sellers, demand) = (&sellers, &demand);
            // Waves of one product per thread; the results of each wave are
            // added in the order of the products.
            for (wave, rngs) in rngs.chunks_mut(threads).enumerate() {
                let results: Vec<GoodTotals> = std::thread::scope(|scope| {
                    let handles: Vec<_> = rngs
                        .iter_mut()
                        .enumerate()
                        .map(|(k, rng)| {
                            let g = wave * threads + k;
                            scope.spawn(move || {
                                let mut state = GoodState::new(sellers);
                                let mut r = GoodTotals::new(I, H);
                                this.good_market(
                                    g, sellers, &mut state, demand, &mut r, &mut **rng,
                                );
                                r
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                        .collect()
                });
                for r in &results {
                    t.add(r);
                }
            }
        } else {
            let view = self.market_view();
            let mut state = GoodState::new(&sellers);
            let mut r = GoodTotals::new(I, H);
            for g in 0..G {
                view.good_market(g, &sellers, &mut state, &demand, &mut r, rng);
                t.add(&r);
            }
        }
        self.update_aggregate_variables(t);
    }

    fn market_view(&self) -> MarketView<'_> {
        MarketView {
            firms: &self.firms,
            prop: &self.prop,
            rotw: &self.rotw,
            gov: &self.gov,
        }
    }

    /// `update_aggregate_variables!`: average prices paid, consumption and
    /// investment of each group, sales as the minimum of supply and demand,
    /// and the households' capital stock.
    fn update_aggregate_variables(&mut self, t: MarketTotals) {
        let MarketTotals {
            I_i,
            DM_i,
            mut P_bar_i,
            mut P_CF_i,
            Q_d_i,
            Q_d_m,
            C_h,
            I_h,
            C_j,
            C_l,
            P_bar_h,
            P_bar_CF_h,
            P_j,
            P_l,
        } = t;
        for i in 0..I_i.len() {
            if I_i[i] > 0.0 {
                P_CF_i[i] /= I_i[i];
            }
            if DM_i[i] > 0.0 {
                P_bar_i[i] /= DM_i[i];
            }
        }
        self.agg.P_bar_h = sum(&C_h) / P_bar_h;
        self.agg.P_bar_CF_h = sum(&I_h) / zero_to_one(P_bar_CF_h);
        self.gov.C_j = C_j;
        self.gov.P_j = C_j / P_j;
        self.rotw.C_l = C_l;
        self.rotw.P_l = C_l / P_l;

        let H_W = self.w_act.len();
        let H_inact = self.w_inact.len();
        let I = self.firms.len();
        let groups = [
            (0, H_W),
            (H_W, H_W + H_inact),
            (H_W + H_inact, H_W + H_inact + I),
        ];
        self.w_act
            .C_h
            .copy_from_slice(&C_h[groups[0].0..groups[0].1]);
        self.w_inact
            .C_h
            .copy_from_slice(&C_h[groups[1].0..groups[1].1]);
        self.firms
            .C_h
            .copy_from_slice(&C_h[groups[2].0..groups[2].1]);
        self.banks.C_h.copy_from_slice(&C_h[groups[2].1..]);
        self.w_act
            .I_h
            .copy_from_slice(&I_h[groups[0].0..groups[0].1]);
        self.w_inact
            .I_h
            .copy_from_slice(&I_h[groups[1].0..groups[1].1]);
        self.firms
            .I_h
            .copy_from_slice(&I_h[groups[2].0..groups[2].1]);
        self.banks.I_h.copy_from_slice(&I_h[groups[2].1..]);

        let f = &mut self.firms;
        for i in 0..I {
            f.Q_i[i] = min(f.Y_i[i] + f.S_i[i], Q_d_i[i]);
        }
        f.I_i = I_i;
        f.Q_d_i = Q_d_i;
        f.P_bar_i = P_bar_i;
        f.DM_i = DM_i;
        f.P_CF_i = P_CF_i;
        let rotw = &mut self.rotw;
        for g in 0..rotw.Y_m.len() {
            rotw.Q_m[g] = min(rotw.Y_m[g], Q_d_m[g]);
        }
        rotw.Q_d_m = Q_d_m;

        for (K, I) in [
            (&mut self.w_act.K_h, &self.w_act.I_h),
            (&mut self.w_inact.K_h, &self.w_inact.I_h),
            (&mut self.firms.K_h, &self.firms.I_h),
            (&mut self.banks.K_h, &self.banks.I_h),
        ] {
            for (k, i) in K.iter_mut().zip(I) {
                *k += i;
            }
        }
    }
}

/// The parts of the model that the market of one product reads. The markets
/// of different products can run in parallel threads on it.
struct MarketView<'a> {
    firms: &'a Firms,
    prop: &'a Properties,
    rotw: &'a RestOfTheWorld,
    gov: &'a Government,
}

impl MarketView<'_> {
    /// `perform_market!(g)`: the firms market and then the retail market of
    /// product g.
    fn good_market<R: Randomness + ?Sized>(
        &self,
        g: usize,
        s: &Sellers,
        state: &mut GoodState,
        demand: &RetailDemand,
        r: &mut GoodTotals,
        rng: &mut R,
    ) {
        state.reset(s, g);
        self.firms_market(g, s, state, r, rng);
        self.retail_market(g, s, state, demand, r, rng);
    }

    /// `perform_firms_market!`: firms buy product g as intermediate input and
    /// as investment good. The first round draws on the sellers' supply. The
    /// second round draws on their spare capacity and only records demand at
    /// the sellers; the buyers receive nothing from it.
    fn firms_market<R: Randomness + ?Sized>(
        &self,
        g: usize,
        s: &Sellers,
        state: &mut GoodState,
        r: &mut GoodTotals,
        rng: &mut R,
    ) {
        let f = &self.firms;
        let (a_sg, b_CF) = (&self.prop.a_sg[g], self.prop.b_CF_g[g]);
        let I = f.len();
        let mut DM_d_ig: Vec<f64> = (0..I)
            .map(|i| a_sg[f.G_i[i]] * f.DM_d_i[i] + b_CF * f.I_d_i[i])
            .collect();
        let mut DM_nominal_ig = vec![0.0; I];
        let mut I_g: Vec<usize> = (0..I).filter(|&i| DM_d_ig[i] > 0.0).collect();

        let S_fg = &mut state.S_fg;
        state.F_g.retain(|&f| S_fg[f] > 0.0);
        let F_g = &state.F_g;
        let mut active = seller_weights(&s.P_f, &s.S_f, F_g);
        while !I_g.is_empty() && !active.is_zero() {
            rng.shuffle(&mut I_g);
            for &i in &I_g {
                let e = rng.sample(&mut active);
                let f = F_g[e];
                if S_fg[f] > DM_d_ig[i] {
                    S_fg[f] -= DM_d_ig[i];
                    DM_nominal_ig[i] += DM_d_ig[i] * s.P_f[f];
                    DM_d_ig[i] = 0.0;
                } else {
                    DM_d_ig[i] -= S_fg[f];
                    DM_nominal_ig[i] += S_fg[f] * s.P_f[f];
                    S_fg[f] = 0.0;
                    active.set_zero(e);
                    if active.is_zero() {
                        break;
                    }
                }
            }
            I_g.retain(|&i| DM_d_ig[i] > 0.0);
        }

        if !I_g.is_empty() {
            let mut DM_d_ig_ = DM_d_ig.clone();
            let F_g_ = s.second_round(g, &state.S_fg_);
            let S_fg_ = &mut state.S_fg_;
            let mut active = seller_weights(&s.P_f, &s.S_f, &F_g_);
            while !I_g.is_empty() && !active.is_zero() {
                rng.shuffle(&mut I_g);
                for &i in &I_g {
                    let e = rng.sample(&mut active);
                    let f = F_g_[e];
                    if S_fg_[f] > DM_d_ig_[i] {
                        S_fg[f] -= DM_d_ig_[i];
                        S_fg_[f] -= DM_d_ig_[i];
                        DM_d_ig_[i] = 0.0;
                    } else {
                        DM_d_ig_[i] -= S_fg_[f];
                        S_fg[f] -= S_fg_[f];
                        S_fg_[f] = 0.0;
                        active.set_zero(e);
                        if active.is_zero() {
                            break;
                        }
                    }
                }
                I_g.retain(|&i| DM_d_ig_[i] > 0.0);
            }
        }

        // Unmet demand reduces investment first, then intermediate inputs.
        for i in 0..I {
            let materials = a_sg[f.G_i[i]] * f.DM_d_i[i];
            let investment = b_CF * f.I_d_i[i];
            let a = materials - pos(DM_d_ig[i] - investment);
            let b = pos(investment - DM_d_ig[i]);
            let c = materials + investment - DM_d_ig[i];
            r.DM_i[i] = a;
            r.I_i[i] = b;
            r.P_bar_i[i] = DM_nominal_ig[i] * a / zero_to_one(c);
            r.P_CF_i[i] = DM_nominal_ig[i] * b / zero_to_one(c);
        }
    }

    /// `perform_retail_market!`: households, foreign consumers and the
    /// government buy product g with nominal budgets, in two rounds as in the
    /// firms market.
    fn retail_market<R: Randomness + ?Sized>(
        &self,
        g: usize,
        s: &Sellers,
        state: &mut GoodState,
        demand: &RetailDemand,
        r: &mut GoodTotals,
        rng: &mut R,
    ) {
        let (C_d_h, I_d_h) = (&demand.C_d_h, &demand.I_d_h);
        let (b_HH, b_CFH) = (demand.b_HH_g[g], demand.b_CFH_g[g]);
        let (c_E, c_G) = (demand.c_E_g[g], demand.c_G_g[g]);
        let (C_d_l, C_d_j) = (&self.rotw.C_d_l, &self.gov.C_d_j);
        let (H, L) = (C_d_h.len(), C_d_l.len());
        let mut C_d_hg: Vec<f64> = (0..H)
            .map(|h| b_HH * C_d_h[h] + b_CFH * I_d_h[h])
            .chain(C_d_l.iter().map(|&c| c_E * c))
            .chain(C_d_j.iter().map(|&c| c_G * c))
            .collect();
        let mut C_real_hg = vec![0.0; C_d_hg.len()];
        let mut H_g: Vec<usize> = (0..C_d_hg.len()).filter(|&h| C_d_hg[h] > 0.0).collect();

        let S_fg = &mut state.S_fg;
        state.F_g.retain(|&f| S_fg[f] > 0.0);
        let F_g = &state.F_g;
        let mut active = seller_weights(&s.P_f, &s.S_f, F_g);
        while !H_g.is_empty() && !active.is_zero() {
            rng.shuffle(&mut H_g);
            for &h in &H_g {
                let e = rng.sample(&mut active);
                let f = F_g[e];
                if S_fg[f] > C_d_hg[h] / s.P_f[f] {
                    S_fg[f] -= C_d_hg[h] / s.P_f[f];
                    C_real_hg[h] += C_d_hg[h] / s.P_f[f];
                    C_d_hg[h] = 0.0;
                } else {
                    C_d_hg[h] -= S_fg[f] * s.P_f[f];
                    C_real_hg[h] += S_fg[f];
                    S_fg[f] = 0.0;
                    active.set_zero(e);
                    if active.is_zero() {
                        break;
                    }
                }
            }
            H_g.retain(|&h| C_d_hg[h] > 0.0);
        }

        if !H_g.is_empty() {
            let mut C_d_hg_ = C_d_hg.clone();
            let F_g_ = s.second_round(g, &state.S_fg_);
            let S_fg_ = &mut state.S_fg_;
            let mut active = seller_weights(&s.P_f, &s.S_f, &F_g_);
            while !H_g.is_empty() && !active.is_zero() {
                rng.shuffle(&mut H_g);
                for &h in &H_g {
                    let e = rng.sample(&mut active);
                    let f = F_g_[e];
                    if S_fg_[f] > C_d_hg_[h] / s.P_f[f] {
                        S_fg[f] -= C_d_hg_[h] / s.P_f[f];
                        S_fg_[f] -= C_d_hg_[h] / s.P_f[f];
                        C_d_hg_[h] = 0.0;
                    } else {
                        C_d_hg_[h] -= S_fg_[f] * s.P_f[f];
                        S_fg[f] -= S_fg_[f];
                        S_fg_[f] = 0.0;
                        active.set_zero(e);
                        if active.is_zero() {
                            break;
                        }
                    }
                }
                H_g.retain(|&h| C_d_hg_[h] > 0.0);
            }
        }

        // Households: unmet demand reduces housing investment first, then
        // consumption.
        let consumption = |h: usize| C_d_h[h] * b_HH;
        let investment = |h: usize| b_CFH * I_d_h[h];
        let b = |h: usize| consumption(h) - pos(C_d_hg[h] - investment(h));
        let c = |h: usize| consumption(h) + investment(h) - C_d_hg[h];
        let d = |h: usize| pos(investment(h) - C_d_hg[h]);
        let sum_a = sum(&C_real_hg[..H]);
        let (sum_b, sum_c, sum_d) = (sum_by(H, b), sum_by(H, c), sum_by(H, d));
        for h in 0..H {
            r.C_h[h] = b(h);
            r.I_h[h] = d(h);
        }
        r.Q_d.clear();
        for &f in &s.by_good[g] {
            r.Q_d.push((f, s.S_f[f] - state.S_fg[f]));
        }
        let budget_j = sum_by(C_d_j.len(), |j| c_G * C_d_j[j]);
        let budget_l = sum_by(C_d_l.len(), |l| c_E * C_d_l[l]);
        r.C_j = budget_j - sum(&C_d_hg[H + L..]);
        r.C_l = budget_l - sum(&C_d_hg[H..H + L]);
        r.P_bar_h = sum_a * sum_b / zero_to_one(sum_c);
        r.P_bar_CF_h = sum_a * sum_d / zero_to_one(sum_c);
        r.P_j = sum(&C_real_hg[H + L..]);
        r.P_l = sum(&C_real_hg[H..H + L]);
    }
}
