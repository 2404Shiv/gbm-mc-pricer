// gbm-mc-pricer: C++17 Monte Carlo GBM pricer for European options
// Features: Greeks (BS closed-form + MC), antithetic & control-variate VR, 1-day P&L/VaR.
// Build: clang++ -O3 -std=c++17 -o pricer src/main.cpp
// Usage: ./pricer '{"S0":100,"K":100,"r":0.02,"sigma":0.2,"T":1.0,"n_paths":1000000,"antithetic":true,"control_variate":true,"seed":42}'

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Config {
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.02;
    double sigma = 0.20;
    double T = 1.0;
    std::uint64_t n_paths = 1'000'000ULL;
    bool antithetic = true;
    bool control_variate = true;
    std::uint64_t seed = 42ULL;
    double bump_rel = 1e-3;          // relative bumps for S0 & sigma
    double bump_t_rel = 1.0/365.0;   // 1 day theta
    bool bump_abs = false;
};

static inline void trim(std::string& s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    size_t i = 0; while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    s.erase(0, i);
}

// very small permissive parser for {"key":value,...}
Config parse_config(const std::string& js, Config base = {}) {
    if (js.empty()) return base;
    std::string s = js; trim(s);
    if (s.front()!='{' || s.back()!='}') return base;
    s = s.substr(1, s.size()-2);
    std::istringstream iss(s);
    auto eat = [&](char c) {
        while (iss && std::isspace(iss.peek())) iss.get();
        if (iss.peek()==c) { iss.get(); return true; }
        return false;
    };
    while (iss) {
        while (iss && std::isspace(iss.peek())) iss.get();
        if (!iss) break;
        if (iss.peek()=='"') {
            iss.get();
            std::string key; std::getline(iss, key, '"');
            while (iss && std::isspace(iss.peek())) iss.get();
            if (!eat(':')) break;
            while (iss && std::isspace(iss.peek())) iss.get();
            if (std::isdigit(iss.peek()) || iss.peek()=='-' || iss.peek()=='+') {
                double v; iss >> v;
                if (key=="S0") base.S0 = v;
                else if (key=="K") base.K = v;
                else if (key=="r") base.r = v;
                else if (key=="sigma") base.sigma = v;
                else if (key=="T") base.T = v;
                else if (key=="n_paths") base.n_paths = static_cast<std::uint64_t>(std::llround(v));
                else if (key=="bump_rel") base.bump_rel = v;
                else if (key=="bump_t_rel") base.bump_t_rel = v;
                else if (key=="seed") base.seed = static_cast<std::uint64_t>(std::llround(v));
            } else if (std::tolower(iss.peek())=='t' || std::tolower(iss.peek())=='f') {
                std::string w; iss >> w;
                bool b = (w=="true" || w=="True" || w=="TRUE");
                if (key=="antithetic") base.antithetic = b;
                else if (key=="control_variate") base.control_variate = b;
                else if (key=="bump_abs") base.bump_abs = b;
            }
            while (iss && std::isspace(iss.peek())) iss.get();
            eat(',');
        } else break;
    }
    return base;
}

static inline double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.39894228040143267793994605993438;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}
static inline double norm_cdf(double x) { return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0))); }

struct BS {
    static inline std::array<double,2> d1d2(double S0, double K, double r, double sigma, double T) {
        double sd = sigma * std::sqrt(T);
        double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / sd;
        double d2 = d1 - sd; return {d1, d2};
    }
    static inline double call(double S0, double K, double r, double sigma, double T) {
        auto [d1,d2] = d1d2(S0,K,r,sigma,T);
        return S0 * norm_cdf(d1) - K * std::exp(-r*T) * norm_cdf(d2);
    }
    static inline double put(double S0, double K, double r, double sigma, double T) {
        auto [d1,d2] = d1d2(S0,K,r,sigma,T);
        return K * std::exp(-r*T) * norm_cdf(-d2) - S0 * norm_cdf(-d1);
    }
    struct Greeks { double delta, gamma, vega, theta, rho; };
    static inline Greeks call_greeks(double S0, double K, double r, double sigma, double T) {
        auto [d1,d2] = d1d2(S0,K,r,sigma,T);
        double Nd1 = norm_cdf(d1), nd1 = norm_pdf(d1);
        double delta = Nd1;
        double gamma = nd1 / (S0 * sigma * std::sqrt(T));
        double vega  = S0 * nd1 * std::sqrt(T);
        double theta = -(S0 * nd1 * sigma)/(2.0*std::sqrt(T)) - r*K*std::exp(-r*T)*norm_cdf(d2);
        double rho   =  K * T * std::exp(-r*T) * norm_cdf(d2);
        return {delta,gamma,vega,theta,rho};
    }
    static inline Greeks put_greeks(double S0, double K, double r, double sigma, double T) {
        auto [d1,d2] = d1d2(S0,K,r,sigma,T);
        double nd1 = norm_pdf(d1);
        double delta = norm_cdf(-d1) - 1.0;
        double gamma = nd1 / (S0 * sigma * std::sqrt(T));
        double vega  = S0 * nd1 * std::sqrt(T);
        double theta = -(S0 * nd1 * sigma)/(2.0*std::sqrt(T)) + r*K*std::exp(-r*T)*norm_cdf(-d2);
        double rho   = -K * T * std::exp(-r*T) * norm_cdf(-d2);
        return {delta,gamma,vega,theta,rho};
    }
};

enum class OptionType { Call, Put };

struct MCResult { double price=0.0, stderror=0.0, control_b=0.0; std::uint64_t n_eff=0; };

class GBMMC {
    Config cfg_; std::mt19937_64 rng_; std::normal_distribution<double> stdn_{0.0,1.0};
public:
    explicit GBMMC(const Config& cfg): cfg_(cfg), rng_(cfg.seed) {}

    MCResult price(OptionType type) {
        const auto N = cfg_.n_paths;
        const bool A = cfg_.antithetic;
        const bool CV = cfg_.control_variate;
        const double S0=cfg_.S0, K=cfg_.K, r=cfg_.r, s=cfg_.sigma, T=cfg_.T;
        const double drift = (r - 0.5*s*s)*T, vol = s*std::sqrt(T), disc = std::exp(-r*T);

        std::vector<double> pay, ST; pay.reserve(N); ST.reserve(N);
        auto payoff = [&](double x){ return (type==OptionType::Call)? std::max(x-K,0.0) : std::max(K-x,0.0); };

        std::uint64_t draws = A ? (N/2) : N;
        for (std::uint64_t i=0;i<draws;++i) {
            double Z = stdn_(rng_);
            double ST1 = S0*std::exp(drift + vol*Z);
            double ST2 = S0*std::exp(drift - vol*Z);
            if (A) { pay.push_back(payoff(ST1)); pay.push_back(payoff(ST2)); ST.push_back(ST1); ST.push_back(ST2); }
            else   { pay.push_back(payoff(ST1)); ST.push_back(ST1); }
        }
        const std::size_t n = pay.size();

        auto mean_std = [&](const std::vector<double>& x){
            double m = std::accumulate(x.begin(),x.end(),0.0)/double(x.size());
            double sq=0.0; for(double v:x){ double d=v-m; sq+=d*d; }
            double var=(x.size()>1)?(sq/(x.size()-1)):0.0;
            return std::pair<double,double>{m,std::sqrt(var)};
        };

        double price=0.0,se=0.0,bopt=0.0;
        if (CV) {
            auto [mY,sY] = mean_std(pay);
            auto [mX,sX] = mean_std(ST);
            double cov=0.0; for (std::size_t i=0;i<n;++i) cov += (pay[i]-mY)*(ST[i]-mX);
            cov /= (n>1?(n-1):1);
            double varX = sX*sX;
            bopt = (varX>0.0)? cov/varX : 0.0;
            const double EX = S0*std::exp(r*T);
            std::vector<double> adj; adj.reserve(n);
            for (std::size_t i=0;i<n;++i) adj.push_back(pay[i] - bopt*(ST[i]-EX));
            auto [mA,sA] = mean_std(adj);
            price = disc*mA; se = disc*(sA/std::sqrt(double(n)));
        } else {
            auto [mP,sP] = mean_std(pay);
            price = disc*mP; se = disc*(sP/std::sqrt(double(n)));
        }
        return {price,se,bopt,(std::uint64_t)n};
    }

    // Pathwise delta
    double mc_delta(OptionType type) {
        const auto N=cfg_.n_paths; const bool A=cfg_.antithetic;
        const double S0=cfg_.S0,K=cfg_.K,r=cfg_.r,s=cfg_.sigma,T=cfg_.T;
        const double drift=(r-0.5*s*s)*T, vol=s*std::sqrt(T), disc=std::exp(-r*T);
        double acc=0.0; std::uint64_t draws=A?(N/2):N;
        for (std::uint64_t i=0;i<draws;++i){
            double Z=stdn_(rng_);
            auto contrib=[&](double ST){ if (type==OptionType::Call){ if (ST>K) acc += (ST/S0); } else { if (ST<K) acc -= (ST/S0); } };
            double ST1=S0*std::exp(drift+vol*Z); contrib(ST1);
            if (A){ double ST2=S0*std::exp(drift-vol*Z); contrib(ST2); }
        }
        return disc * (acc / double(N));
    }

    struct GreekSet { double delta,gamma,vega,theta,rho; };
    GreekSet mc_greeks_bump(OptionType type) {
        auto base_seed=cfg_.seed;
        auto price_with=[&](double S0,double K,double r,double s,double T){
            Config c=cfg_; c.S0=S0;c.K=K;c.r=r;c.sigma=s;c.T=T;c.seed=base_seed;
            GBMMC tmp(c); return tmp.price(type).price;
        };
        double eps=cfg_.bump_rel;
        double S0=cfg_.S0,r=cfg_.r,s=cfg_.sigma,T=cfg_.T;
        double dS=cfg_.bump_abs?eps:(S0*eps);
        double dv=cfg_.bump_abs?eps:(s*eps);
        double dt=cfg_.bump_abs?eps:(cfg_.bump_t_rel);
        double P0=price_with(S0,cfg_.K,r,s,T);
        double Pup=price_with(S0+dS,cfg_.K,r,s,T), Pdn=price_with(S0-dS,cfg_.K,r,s,T);
        double Pvu=price_with(S0,cfg_.K,r,s+dv,T), Pvd=price_with(S0,cfg_.K,r,s-dv,T);
        double Pt =price_with(S0,cfg_.K,r,s,std::max(1e-8,T-dt));
        double Pru=price_with(S0,cfg_.K,r+eps,s,T);
        double delta=(Pup-Pdn)/(2.0*dS);
        double gamma=(Pup-2.0*P0+Pdn)/(dS*dS);
        double vega =(Pvu-Pvd)/(2.0*dv);
        double theta=(Pt-P0)/(-dt);
        double rho  =(Pru-P0)/(eps);
        return {delta,gamma,vega,theta,rho};
    }

    struct VaRPanel { double mean,stdev,VaR95,VaR99,ES95; };
    VaRPanel pnl_var_1d(OptionType type, std::uint64_t n_sims=200'000, std::uint64_t seed=123) {
        std::mt19937_64 r2(seed); std::normal_distribution<double> Z(0.0,1.0);
        const double S0=cfg_.S0,K=cfg_.K,r=cfg_.r,s=cfg_.sigma; double T=cfg_.T;
        const double dt=1.0/252.0, drift=(r-0.5*s*s)*dt, vol=s*std::sqrt(dt);
        auto price_bs=[&](double S,double Tleft){ return (type==OptionType::Call)? BS::call(S,K,r,s,Tleft) : BS::put(S,K,r,s,Tleft); };
        const double P0=price_bs(S0,T);
        std::vector<double> pnl; pnl.reserve(n_sims);
        for (std::uint64_t i=0;i<n_sims;++i){
            double s1=S0*std::exp(drift+vol*Z(r2));
            double T1=std::max(1e-8,T-dt);
            double P1=price_bs(s1,T1);
            pnl.push_back(P1-P0);
        }
        auto mean_std=[&](const std::vector<double>& x){
            double m=std::accumulate(x.begin(),x.end(),0.0)/double(x.size());
            double sq=0.0; for(double v:x){ double d=v-m; sq+=d*d; }
            double var=(x.size()>1)?(sq/(x.size()-1)):0.0;
            return std::pair<double,double>{m,std::sqrt(var)};
        };
        auto [m,sd]=mean_std(pnl);
        std::vector<double> sorted=pnl; std::sort(sorted.begin(),sorted.end());
        auto q=[&](double a){ size_t idx=size_t(std::floor(a*(sorted.size()-1))); return sorted[idx]; };
        double VaR95 = -q(0.05), VaR99 = -q(0.01);
        size_t cut = std::max<size_t>(1,size_t(std::floor(0.05*sorted.size())));
        double es_sum=0.0; for(size_t i=0;i<cut;++i) es_sum+=sorted[i];
        double ES95 = -(es_sum/double(cut));
        return {m,sd,VaR95,VaR99,ES95};
    }
};

static void banner(){
    std::cout << "GBM Monte Carlo Pricer (C++17) — Greeks, antithetic + control-variate, and P&L/VaR.\n";
}

int main(int argc, char** argv){
    banner();
    Config cfg; if (argc>=2) cfg = parse_config(argv[1], cfg);
    GBMMC engine(cfg);

    double bs_call=BS::call(cfg.S0,cfg.K,cfg.r,cfg.sigma,cfg.T);
    double bs_put =BS::put (cfg.S0,cfg.K,cfg.r,cfg.sigma,cfg.T);
    auto cg=BS::call_greeks(cfg.S0,cfg.K,cfg.r,cfg.sigma,cfg.T);
    auto pg=BS::put_greeks (cfg.S0,cfg.K,cfg.r,cfg.sigma,cfg.T);

    MCResult mc_call=engine.price(OptionType::Call);
    MCResult mc_put =engine.price(OptionType::Put);

    double mc_delta_call=engine.mc_delta(OptionType::Call);
    double mc_delta_put =engine.mc_delta(OptionType::Put);
    auto mcg_call=engine.mc_greeks_bump(OptionType::Call);
    auto mcg_put =engine.mc_greeks_bump(OptionType::Put);

    auto var_call=engine.pnl_var_1d(OptionType::Call);
    auto var_put =engine.pnl_var_1d(OptionType::Put);

    std::cout.setf(std::ios::fixed); std::cout<<std::setprecision(6);
    std::cout << "\nConfig: S0="<<cfg.S0<<" K="<<cfg.K<<" r="<<cfg.r
              << " sigma="<<cfg.sigma<<" T="<<cfg.T<<" n_paths="<<cfg.n_paths
              << " antithetic="<<(cfg.antithetic?"true":"false")
              << " control_variate="<<(cfg.control_variate?"true":"false")
              << " seed="<<cfg.seed << "\n";

    std::cout << "\n=== Black-Scholes (closed form) ===\n";
    std::cout << "Call = " << bs_call << " | Put = " << bs_put << "\n";
    std::cout << "Call Greeks  [Δ="<<cg.delta<<", Γ="<<cg.gamma<<", Vega="<<cg.vega
              << ", Θ="<<cg.theta<<", ρ="<<cg.rho<<"]\n";
    std::cout << "Put  Greeks  [Δ="<<pg.delta<<", Γ="<<pg.gamma<<", Vega="<<pg.vega
              << ", Θ="<<pg.theta<<", ρ="<<pg.rho<<"]\n";

    std::cout << "\n=== Monte Carlo ===\n";
    std::cout << "Call_MC = " << mc_call.price << "  (SE=" << mc_call.stderror
              << ", b*=" << mc_call.control_b << ", n=" << mc_call.n_eff << ")\n";
    std::cout << "Put_MC  = " << mc_put.price  << "  (SE=" << mc_put.stderror
              << ", b*=" << mc_put.control_b  << ", n=" << mc_put.n_eff  << ")\n";

    std::cout << "\nMC Delta (pathwise): call="<<mc_delta_call<<", put="<<mc_delta_put<<"\n";
    std::cout << "MC Greeks (bump, CRN) — Call [Δ="<<mcg_call.delta<<", Γ="<<mcg_call.gamma
              << ", Vega="<<mcg_call.vega<<", Θ="<<mcg_call.theta<<", ρ="<<mcg_call.rho<<"]\n";
    std::cout << "MC Greeks (bump, CRN) — Put  [Δ="<<mcg_put.delta<<", Γ="<<mcg_put.gamma
              << ", Vega="<<mcg_put.vega<<", Θ="<<mcg_put.theta<<", ρ="<<mcg_put.rho<<"]\n";

    std::cout << "\n=== 1-Day P&L / VaR (revaluation) ===\n";
    std::cout << "Call: mean="<<var_call.mean<<" stdev="<<var_call.stdev
              << "  VaR95="<<var_call.VaR95<<"  VaR99="<<var_call.VaR99
              << "  ES95="<<var_call.ES95<<"\n";
    std::cout << "Put : mean="<<var_put.mean<<" stdev="<<var_put.stdev
              << "  VaR95="<<var_put.VaR95<<"  VaR99="<<var_put.VaR99
              << "  ES95="<<var_put.ES95<<"\n\n";

    std::cout << "Tip: pass a JSON string to override params.\n";
    return 0;
}
