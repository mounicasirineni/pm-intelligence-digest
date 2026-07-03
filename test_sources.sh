#!/bin/bash

echo "================================================================"
echo "SOURCE URL HEALTH CHECK"
echo "================================================================"
echo ""

# Results buckets
declare -a OK=()
declare -a REDIRECT=()
declare -a BLOCKED=()
declare -a BROKEN=()
declare -a ENV_VARS=()

check() {
  local name="$1"
  local url="$2"

  # Skip env: URLs
  if [[ "$url" == env:* ]]; then
    ENV_VARS+=("$name → $url")
    return
  fi

  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 --location "$url" 2>/dev/null)

  if [[ "$code" == "200" ]]; then
    OK+=("✓ $name")
  elif [[ "$code" == "301" || "$code" == "302" || "$code" == "303" || "$code" == "307" || "$code" == "308" ]]; then
    local final
    final=$(curl -s -o /dev/null -w "%{url_effective}" --max-time 10 --location "$url" 2>/dev/null)
    REDIRECT+=("↪ $name ($code) → $final")
  elif [[ "$code" == "403" || "$code" == "401" || "$code" == "429" ]]; then
    BLOCKED+=("✗ $name ($code) — blocked/auth required")
  elif [[ "$code" == "404" || "$code" == "410" ]]; then
    BROKEN+=("✗ $name ($code) — URL not found")
  elif [[ "$code" == "000" ]]; then
    BROKEN+=("✗ $name (timeout/unreachable)")
  else
    BROKEN+=("? $name ($code) — unexpected status")
  fi
}

# --- Run checks ---
check "Andreessen Horowitz Podcast"   "https://feeds.simplecast.com/JGE3yC0V"
check "Platformer"                    "https://www.platformer.news/feed/"
check "MIT Technology Review"         "https://www.technologyreview.com/feed/"
check "Import AI (Jack Clark)"        "https://importai.substack.com/feed"
check "Shreyas Doshi"                 "https://shreyasdoshi.substack.com/feed"
check "SVPG (Marty Cagan)"            "https://www.svpg.com/feed/"
check "TechCrunch"                    "https://techcrunch.com/feed/"
check "Y Combinator Blog"             "https://ycombinator.com/blog/rss"
check "Hacker News Frontpage"         "https://hnrss.org/frontpage"
check "Google Blog"                   "https://blog.google/rss/"
check "Meta Engineering"              "https://engineering.fb.com/feed/"
check "Microsoft Research Blog"       "https://www.microsoft.com/en-us/research/feed/"
check "Simon Willison's Blog"         "https://simonwillison.net/atom/everything/"
check "OpenAI News"                   "https://openai.com/news/rss.xml"
check "Rest of World"                 "https://restofworld.org/feed/"
check "Benedict Evans"                "https://www.ben-evans.com/benedictevans/rss.xml"
check "AI Snake Oil"                  "https://aisnakeoil.substack.com/feed"
check "The Verge"                     "https://www.theverge.com/rss/index.xml"
check "Apple Newsroom"                "https://www.apple.com/newsroom/rss-feed.rss"
check "Amazon News"                   "https://www.aboutamazon.com/news/rss"
check "NVIDIA Blog"                   "https://blogs.nvidia.com/feed/"
check "Lenny's Podcast"               "env:LENNYS_PODCAST_RSS"
check "Stratechery (Free)"            "https://stratechery.com/feed/"
check "Nielsen Norman Group"          "https://www.nngroup.com/feed/rss/"
check "UX Collective"                 "https://uxdesign.cc/feed"
check "Politico Tech"                 "https://rss.politico.com/technology.xml"
check "EFF"                           "https://www.eff.org/rss/updates.xml"
check "Axios Business"                "https://www.axios.com/feeds/feed.rss"
check "Hard Fork"                     "https://feeds.simplecast.com/l2i9YnTd"
check "Dwarkesh Podcast"              "https://dwarkeshpatel.substack.com/feed"
check "Pivot"                         "https://feeds.megaphone.fm/pivot"
check "Lenny's Newsletter"            "env:LENNYS_NEWSLETTER_RSS"
check "How I AI (Claire Vo)"          "env:HOW_I_AI_RSS"
check "Sifted"                        "https://sifted.eu/feed"
check "YourStory"                     "https://yourstory.com/feed"
check "Netflix Tech Blog"             "https://netflixtechblog.com/feed"
check "Anthropic News"                "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml"

# --- Report ---
echo "✓ WORKING (${#OK[@]})"
echo "----------------------------------------------------------------"
for s in "${OK[@]}"; do echo "  $s"; done

echo ""
echo "↪ REDIRECTS — URL may be stale (${#REDIRECT[@]})"
echo "----------------------------------------------------------------"
for s in "${REDIRECT[@]}"; do echo "  $s"; done

echo ""
echo "✗ BLOCKED / AUTH (${#BLOCKED[@]})"
echo "----------------------------------------------------------------"
for s in "${BLOCKED[@]}"; do echo "  $s"; done

echo ""
echo "✗ BROKEN / NOT FOUND (${#BROKEN[@]})"
echo "----------------------------------------------------------------"
for s in "${BROKEN[@]}"; do echo "  $s"; done

echo ""
echo "~ ENV VAR URLs — not testable here (${#ENV_VARS[@]})"
echo "----------------------------------------------------------------"
for s in "${ENV_VARS[@]}"; do echo "  $s"; done

echo ""
echo "================================================================"
echo "SUMMARY: ${#OK[@]} ok · ${#REDIRECT[@]} redirects · ${#BLOCKED[@]} blocked · ${#BROKEN[@]} broken · ${#ENV_VARS[@]} env vars"
echo "================================================================"
