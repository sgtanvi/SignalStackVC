import React, { useState } from 'react';
import { Globe, Squirrel, TrendingUp, Search, Sparkle, Users, Filter, Tag, ExternalLink, ListCheck, Rocket, 
  Github, Linkedin, Star, Eye, Bell 
      } from 'lucide-react';

const SignalStackVC = () => {
  const [searchUrl, setSearchUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [watchlist, setWatchlist] = useState([]);
  const [activeTab, setActiveTab] = useState('search');
  const [selectedStartup, setSelectedStartup] = useState(null);

  //mock data for boilerplate / demo
  const mockStartups = [
    {
      id: 1,
      name: "FlowAI",
      url: "https://flow.ai",
      description: "AI-powered workflow automation platform that helps teams eliminate repetitive tasks and focus on high-impact work.",
      category: "AI/ML",
      stage: "Series A",
      funding: "$12M",
      team: 24,
      founded: "2022",
      traction: {
        productHunt: { votes: 1247, rank: "#2 Product of the Day" },
        github: { stars: 3400, commits: 892 },
        linkedin: { employees: 24, growth: "+35% (6mo)" }
      },
      signals: {
        hot: true,
        funding: "Recent Series A",
        hiring: "5 new hires this month",
        product: "Major feature launch"
      },
      notes: "Strong product-market fit in enterprise automation",
      tags: ["AI", "Enterprise", "Automation"]
    },
    {
      id: 2,
      name: "CarbonScale",
      url: "https://carbonscale.io",
      description: "Carbon accounting and ESG reporting platform for mid-market companies looking to track and reduce their environmental impact.",
      category: "Climate Tech",
      stage: "Seed",
      funding: "$3.2M",
      team: 12,
      founded: "2023",
      traction: {
        productHunt: { votes: 892, rank: "#5 Product of the Day" },
        github: { stars: 156, commits: 234 },
        linkedin: { employees: 12, growth: "+20% (3mo)" }
      },
      signals: {
        hot: false,
        funding: "Seed closed Q1 2024",
        hiring: "2 new hires this month",
        product: "API integrations live"
      },
      notes: "Interesting climate tech play, strong founder background",
      tags: ["Climate", "SaaS", "ESG"]
    }
  ];

  const handleSearch = async () => {
    if (!searchUrl.trim()) return;
    
    setIsLoading(true);
    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      setSelectedStartup(mockStartups[0]);
      setActiveTab('profile');
    }, 2000);
  };

  const addToWatchlist = (startup) => {
    if (!watchlist.find(item => item.id === startup.id)) {
      setWatchlist([...watchlist, startup]);
    }
  };

  const SignalBadge = ({ type, text, hot = false }) => (
    <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
      hot 
        ? 'bg-orange-100 text-orange-800 border border-orange-200' 
        : 'bg-blue-50 text-blue-700 border border-blue-200'
    }`}>
      {hot && <TrendingUp className="w-3 h-3 mr-1" />}
      {text}
    </div>
  );

  const SearchTab = () => (
    <div className = "min-h-screen flex items-start pt-[20vh]">
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          SignalStackVC
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          All the startup signals, in one place. So you can stop hunting and start investing.
        </p>
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <div className="flex gap-4 mb-6">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Paste startup URL or name..."
                value={searchUrl}
                onChange={(e) => setSearchUrl(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 flex items-center gap-2">
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <Search className="w-5 h-5" />
              )}
              {isLoading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <Sparkle className="w-4 h-4 text-blue-600" />
              Analyze Startup
            </div>
            <div className="flex items-center gap-2">
              <Users className="w-4 h-4 text-blue-600" />
              Analyze Batch
            </div>
            <div className="flex items-center gap-2">
              <ListCheck className="w-4 h-4 text-blue-600" />
              List Startups
            </div>
            <div className="flex items-center gap-2">
              <Squirrel className="w-4 h-4 text-blue-600" />
              Get Startup
            </div>
          </div>
        </div>
      </div>
    </div>
    </div>
  );

  const ProfileTab = () => {
    if (!selectedStartup) return <div>No startup selected</div>;
    
    return (
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          {/* Header */}
          <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                  {selectedStartup.name.charAt(0)}
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">{selectedStartup.name}</h2>
                  <div className="flex items-center gap-2 text-gray-600">
                    <Globe className="w-4 h-4" />
                    <a href={selectedStartup.url} className="text-blue-600 hover:text-blue-700">
                      {selectedStartup.url}
                    </a>
                    <ExternalLink className="w-4 h-4" />
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {selectedStartup.signals.hot && (
                  <SignalBadge type="hot" text="🔥 Hot Signal" hot={true} />
                )}
                <button
                  onClick={() => addToWatchlist(selectedStartup)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
                >
                  <Star className="w-4 h-4" />
                  Add to Watchlist
                </button>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Info */}
              <div className="lg:col-span-2 space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">What they do</h3>
                  <p className="text-gray-700 leading-relaxed">{selectedStartup.description}</p>
                  <div className="flex items-center gap-2 mt-3">
                    {selectedStartup.tags.map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Traction Signals</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Rocket className="w-5 h-5 text-orange-600" />
                        <span className="font-medium text-gray-900">Product Hunt</span>
                      </div>
                      <div className="text-2xl font-bold text-gray-900">{selectedStartup.traction.productHunt.votes}</div>
                      <div className="text-sm text-gray-600">{selectedStartup.traction.productHunt.rank}</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Github className="w-5 h-5 text-gray-700" />
                        <span className="font-medium text-gray-900">GitHub</span>
                      </div>
                      <div className="text-2xl font-bold text-gray-900">{selectedStartup.traction.github.stars}</div>
                      <div className="text-sm text-gray-600">{selectedStartup.traction.github.commits} commits</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Linkedin className="w-5 h-5 text-blue-600" />
                        <span className="font-medium text-gray-900">LinkedIn</span>
                      </div>
                      <div className="text-2xl font-bold text-gray-900">{selectedStartup.traction.linkedin.employees}</div>
                      <div className="text-sm text-green-600">{selectedStartup.traction.linkedin.growth}</div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Live Signals</h3>
                  <div className="space-y-2">
                    <SignalBadge type="funding" text={selectedStartup.signals.funding} />
                    <SignalBadge type="hiring" text={selectedStartup.signals.hiring} />
                    <SignalBadge type="product" text={selectedStartup.signals.product} />
                  </div>
                </div>
              </div>

              {/* Sidebar */}
              <div className="space-y-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-3">Quick Stats</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Stage</span>
                      <span className="font-medium text-gray-900">{selectedStartup.stage}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Funding</span>
                      <span className="font-medium text-gray-900">{selectedStartup.funding}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Team Size</span>
                      <span className="font-medium text-gray-900">{selectedStartup.team}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Founded</span>
                      <span className="font-medium text-gray-900">{selectedStartup.founded}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Category</span>
                      <span className="font-medium text-gray-900">{selectedStartup.category}</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-3">Your Notes</h3>
                  <textarea
                    placeholder="Add your analysis and notes..."
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900 h-24 resize-none"
                    defaultValue={selectedStartup.notes}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const WatchlistTab = () => (
    <div className="max-w-6xl mx-auto">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Your Watchlist</h2>
          <p className="text-gray-600">Track and monitor your portfolio prospects</p>
        </div>
        <div className="p-6">
          {watchlist.length === 0 ? (
            <div className="text-center py-12">
              <Eye className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No startups in your watchlist yet</p>
              <p className="text-sm text-gray-500">Add companies to track their signals and updates</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {watchlist.map((startup) => (
                <div key={startup.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                      {startup.name.charAt(0)}
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{startup.name}</h3>
                      <p className="text-sm text-gray-600">{startup.category}</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-700 mb-3 line-clamp-2">{startup.description}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-900">{startup.funding}</span>
                    <button
                      onClick={() => {
                        setSelectedStartup(startup);
                        setActiveTab('profile');
                      }}
                      className="text-sm text-blue-600 hover:text-blue-700"
                    >
                      View →
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-8">
              <div className="font-bold text-xl text-gray-900">SignalStackVC</div>
              <div className="flex space-x-8">
                <button
                  onClick={() => setActiveTab('search')}
                  className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeTab === 'search'
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Search className="w-4 h-4 inline mr-2" />
                  Search
                </button>
                <button
                  onClick={() => setActiveTab('profile')}
                  className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeTab === 'profile'
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Users className="w-4 h-4 inline mr-2" />
                  Profile
                </button>
                <button
                  onClick={() => setActiveTab('watchlist')}
                  className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    activeTab === 'watchlist'
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Star className="w-4 h-4 inline mr-2" />
                  Watchlist ({watchlist.length})
                </button>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <button className="p-2 text-gray-600 hover:text-gray-900">
                <Bell className="w-5 h-5" />
              </button>
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-medium text-sm">
                VC
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="py-8 px-4 sm:px-6 lg:px-8">
        {activeTab === 'search' && <SearchTab />}
        {activeTab === 'profile' && <ProfileTab />}
        {activeTab === 'watchlist' && <WatchlistTab />}
      </main>
    </div>
  );
};

export default SignalStackVC;