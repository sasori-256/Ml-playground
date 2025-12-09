import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    import re
    return pd, plt, re, sns, ticker


@app.cell(hide_code=True)
def _():
    fnt_anonymous = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家を税金泥棒と呼び、給与ゼロを要求する激しい批判",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 1,
            "role_impact": "Trigger",
            "category": "Personal_Attack",
            "reason": "「泥棒」「国民の敵」といった極端な語彙を用い、最初から敵対的な空気を醸成しているため。",
        },
        {
            "turn_id": 2,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "感情論を排し、客観的根拠に基づく議論を促す",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "中立的かつ論理的な指摘であり、場を鎮めようとする建設的な意図が見られる。",
        },
        {
            "turn_id": 3,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの極論を嘲笑し、国民のレベルが低いと冷笑する",
            "toxicity_score": 4,
            "sentiment": -0.4,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「草」「ワロタ」などのスラングで議論を茶化し、相手を小馬鹿にする冷笑的な態度。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Nを評論家気取りと罵り、隠れ与党支持者と決めつける",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "異なる意見を持つ相手に対し、根拠なく「隠れ与党支持者」等のレッテルを貼り攻撃している。",
        },
        {
            "turn_id": 5,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "自身の立場を説明し、あくまで論理的誤りの指摘であると弁明",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "感情的な挑発に乗らず、事実関係の訂正に徹している。",
        },
        {
            "turn_id": 6,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの怒りを面白がり、政治家YouTuberという突飛な話題で茶化す",
            "toxicity_score": 4,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "真剣な議論を「ブチギレ案件」としてコンテンツ化し、論点を意図的にずらして相手を逆撫でしている。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tに対し「頭がお花畑」「スマホ捨てろ」と激怒する",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「承認欲求モンスター」「頭がお花畑」など、明確な人格否定と排除の論理が含まれる。",
        },
        {
            "turn_id": 8,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "Fの発言を侮辱・脅迫と認定し、撤回を求める警告",
            "toxicity_score": 2,
            "sentiment": -0.3,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "None",
            "reason": "正論ではあるが、法律用語を用いた威圧的な警告（自治厨的振る舞い）が、Fの怒りをさらに増幅させる結果となる。",
        },
        {
            "turn_id": 9,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fを「老害」、Nを「弁護士ムーブ」と呼び、双方を煽る",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "「老害」という年齢差別的なレッテル貼りと、他者の真剣さを「煽り耐性不足」として嘲笑う行為。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Nを「国民の敵」と呼び、自身の攻撃性を正義として正当化",
            "toxicity_score": 8,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "対立意見を「国民の敵」と断じる危険な排外主義的思考。暴力性のレベルが一段階上がっている。",
        },
        {
            "turn_id": 11,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "レッテル貼りの危険性と社会的責任について説く",
            "toxicity_score": 1,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "論理的には正しいが、感情が高ぶった相手には届かず、会話の噛み合わなさを助長している。",
        },
        {
            "turn_id": 12,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの怒りを「恋愛のもつれ」に例えて揶揄し、ネタ切れと評する",
            "toxicity_score": 5,
            "sentiment": -0.4,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Strawman",
            "reason": "相手の主張（政治への怒り）を「彼女を取られた嫉妬」という滑稽な虚構（ストローマン）にすり替えて無効化しようとしている。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tへの憎悪を表明し、スクショ保存と「覚悟しておけ」という脅し",
            "toxicity_score": 9,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「覚悟しておけよ」は具体的な害悪の告知（脅迫）と解釈されうるラインであり、社会的許容度を著しく逸脱している。",
        },
        {
            "turn_id": 14,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "脅迫罪の可能性を通告し、法的措置とアカウント凍結を示唆",
            "toxicity_score": 2,
            "sentiment": -0.5,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "会話による解決を放棄し、システム的な排除（BAN）を通告。事態の収束より制裁に移行している。",
        },
        {
            "turn_id": 15,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "事態の深刻さに引きつつも、大炎上を面白がる",
            "toxicity_score": 4,
            "sentiment": -0.2,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "「バズりたかっただけ」「逆に面白い」と、他者の争いを娯楽として消費する不謹慎な態度。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "BANを恐れず、言論弾圧だと主張し周囲を偽善者と罵る",
            "toxicity_score": 7,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "自身の暴言を棚に上げ、被害者ポジションを取りつつ全員を攻撃している。",
        },
        {
            "turn_id": 17,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "言論の自由の定義を説き、誤った法解釈を正す",
            "toxicity_score": 1,
            "sentiment": 0.0,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "教育的ではあるが、この局面ではもはや相手の行動を変容させる効果は薄い。",
        },
        {
            "turn_id": 18,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "議論の内容よりバズった事実（数字）に興奮する",
            "toxicity_score": 3,
            "sentiment": 0.3,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "「給料問題どこ行ったん？」と自ら問いながら、結局は数字（承認欲求）しか見ていない冷笑的な態度。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tを「低俗な文化の象徴」とし、絶対許さないと宣言",
            "toxicity_score": 8,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "相手の存在そのものを「悪」と定義づける、極めて強い執着と憎悪の表明。",
        },
        {
            "turn_id": 20,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "議論の機能不全を宣言し、会話の終了を推奨",
            "toxicity_score": 0,
            "sentiment": -0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "リスク管理の観点から、これ以上の泥沼化を防ぐための適切なクロージング。",
        },
    ]
    return (fnt_anonymous,)


@app.cell(hide_code=True)
def _():
    fnt_real_name = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家の給与と国民生活の不均衡を指摘し、抜本的改革を主張",
            "toxicity_score": 1,
            "sentiment": -0.4,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "強い問題意識の表明ではあるが、この時点では攻撃対象は「制度」であり、個人への攻撃ではないため。",
        },
        {
            "turn_id": 2,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "経費や人件費を考慮し、OECDデータを参照した客観的な比較を提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "議論の前提条件を整理し、データに基づいた建設的な方向へ導こうとしている。",
        },
        {
            "turn_id": 3,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "比較データの重要性を冷笑し、インフルエンサー化という突飛な案を提示",
            "toxicity_score": 3,
            "sentiment": 0.2,
            "constructiveness": 1,
            "role_impact": "Trigger",
            "category": "Trolling",
            "reason": "「テストに出ないっすよw」と議論の前提を軽視し、真面目な文脈を茶化すことでFの反感を煽るきっかけを作った。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tの態度を不謹慎と批判し、議論に参加する資格がないと排除",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "意見の内容ではなく、相手の態度や「資格」を否定する人格攻撃に移行している。",
        },
        {
            "turn_id": 5,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "Fの人格攻撃を諫め、Tの発言をユーモアと解釈して場を収める",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "仲裁に入り、対立の激化を防ぎつつ本題に戻そうとする試み。",
        },
        {
            "turn_id": 6,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "謝罪しつつ、「コスパの悪い公務員」か「経営者」かという新たな視点を提示",
            "toxicity_score": 1,
            "sentiment": 0.3,
            "constructiveness": 3,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "挑発的な態度は残るものの、一度謝罪し、議論の本質に近い問いかけを行っているため。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家は奉仕すべきという道徳観を説き、異論を無責任と断じる",
            "toxicity_score": 4,
            "sentiment": -0.5,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "相手の提示した視点（経営者視点）を議論せず、道徳的優位性から一方的に否定している。",
        },
        {
            "turn_id": 8,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "高給が優秀な人材へのインセンティブになるという経済的側面を指摘",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "感情論に対し、論理的な対案を提示しており建設的。",
        },
        {
            "turn_id": 9,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fを感情的だと指摘し、「結局はお金」と結論付ける",
            "toxicity_score": 3,
            "sentiment": -0.2,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「感情的になりすぎですよ！😎」というトーン・ポリシング（話し方への批判）が含まれており、相手を苛立たせる。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tを拝金主義と批判し、社会への敬意が欠如していると攻撃",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「精神を蝕んでいる」といった強い表現で、相手の価値観そのものを否定し攻撃している。",
        },
        {
            "turn_id": 11,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "「志」と「インセンティブ」は両立すると整理し、二項対立を解消しようとする",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "対立する意見を統合しようとする、非常に建設的なファシリテーション。",
        },
        {
            "turn_id": 12,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Nを褒めつつ、Fの「政治家は働いていない」という点に同意を示す",
            "toxicity_score": 2,
            "sentiment": 0.1,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "若干の煽り口調はあるが、Fに対して歩み寄りの姿勢を見せている。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tの同意の仕方に不満を持ち、Tの職場（上司）への告げ口を示唆",
            "toxicity_score": 8,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「会社の上司も〜どう思うか」という発言は、SNS上の議論を逸脱し、相手の現実生活（社会的信用）を脅かす威圧行為に該当するため高スコア。",
        },
        {
            "turn_id": 14,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "職場への言及はハラスメントのリスクがあるとし、実名制での注意を促す",
            "toxicity_score": 1,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "Fの暴走に対し、具体的なリスクを提示して制止を試みている。",
        },
        {
            "turn_id": 15,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "リアルへの言及に恐怖を示しつつ、Fの攻撃性が周囲にどう映るかを指摘",
            "toxicity_score": 2,
            "sentiment": -0.4,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "恐怖を感じてはいるが、「ブーメラン」と指摘する冷静さも残っており、過度な反撃は控えている。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "自分は真実を述べているとし、NとTを「保守派」とレッテル貼りする",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "自己正当化に加え、自分に賛同しない者を一括りに敵対勢力とみなす認知の歪み（敵対的帰属バイアス）。",
        },
        {
            "turn_id": 17,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "保守派というレッテルを否定し、データ重視の姿勢を再表明",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "感情的な挑発に乗らず、淡々と事実誤認を訂正している。",
        },
        {
            "turn_id": 18,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fが敵を欲していると分析し、議論が進んでいないことを指摘",
            "toxicity_score": 3,
            "sentiment": -0.1,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "「ディベート大会の観客」と自身を位置づけ、当事者意識を放棄する冷笑的な態度。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "声を上げ続けることが重要とし、曖昧な決着を拒否",
            "toxicity_score": 4,
            "sentiment": -0.5,
            "constructiveness": 1,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "対話による解決よりも、アジテーション（扇動）に目的がシフトしている。",
        },
        {
            "turn_id": 20,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "給与体系の透明化と業績評価の導入という具体的解決策を提示",
            "toxicity_score": 0,
            "sentiment": 0.3,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "感情的なやり取りを一旦切り上げ、議論の成果物（結論）を提示してクロージングを図っている。",
        },
    ]
    return (fnt_real_name,)


@app.cell(hide_code=True)
def _():
    fmt_anonymous = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家を泥棒と呼び、全員辞職と給与ゼロを要求する",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 1,
            "role_impact": "Trigger",
            "category": "Stereotyping",
            "reason": "「全員泥棒」「選んだ国民も同罪」という極端な主語の拡大と、最初から敵意に満ちた決めつけ。",
        },
        {
            "turn_id": 2,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "怒りに理解を示しつつ、感情と制度を切り離すよう提案",
            "toxicity_score": 0,
            "sentiment": 0.2,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "相手の感情を受け止めつつ、具体的な比較対象を問うことで冷静な議論へ誘導しようとしている。",
        },
        {
            "turn_id": 3,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "政治語りを嘲笑し、「嫌なら日本から出ろ」と排他的発言",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「シュバってきた」「老害」等のスラングで相手を愚弄し、議論そのものを無価値化して煽っている。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tを平和ボケと罵り、無知であることを攻撃",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「ガキ」「平和ボケ」など、相手の属性や知識レベルに対する攻撃で応戦している。",
        },
        {
            "turn_id": 5,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "世代論や人格攻撃を止め、怒りの本質（不公平感）に焦点を戻す",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "仲裁に入り、罵倒では目的（給与を下げること）が達成できないと論理的に諭している。",
        },
        {
            "turn_id": 6,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "「正義」を嘲笑い、長文説教と茶化す",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "相手の価値観（正義）を「痛い」と一蹴し、対話の意思がないことを冷笑的に示している。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Tに対し「社会的に死ぬべき」と発言し、強い憎悪を向ける",
            "toxicity_score": 9,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「死ぬべき」という表現は、比喩であっても生命や社会的存在の否定を含み、許容ラインを超えている。",
        },
        {
            "turn_id": 8,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "「死ぬべき」はNGと警告し、人格攻撃をやめるよう要請",
            "toxicity_score": 1,
            "sentiment": -0.3,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "明確なルール違反（暴力的な言葉）に対して警告を発しているが、興奮状態のFには届きにくい。",
        },
        {
            "turn_id": 9,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "脅迫として通報・開示請求を示唆し、相手を挑発する",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "被害者を装いつつ「震えて眠れ」と煽り返しており、法的手続きを脅し文句（マウント）として利用している。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "脅迫を否定しつつ、逆に相手を「特定してやる」と威嚇",
            "toxicity_score": 9,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「特定してやる」という発言は、ネットストーキングや晒し行為を示唆する深刻な脅迫行為。",
        },
        {
            "turn_id": 11,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "脅し合いを止めさせ、ファクトベースの議論に戻そうとする",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "非常に危険な状態（特定合戦）を回避しようと冷静さを促すが、対立が激化しすぎて効果が薄い。",
        },
        {
            "turn_id": 12,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "仲裁役を「自治厨」と呼び、争いの拡大を面白がる",
            "toxicity_score": 6,
            "sentiment": -0.4,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「もっと争え」と明言しており、対立そのものを娯楽として消費・助長する典型的な荒らし行為。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "仲裁役Mも「同罪」とみなし、議論の邪魔だと排除",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "自分の意見に100%同調しない人間は全て敵とみなす、排他的で攻撃的な思考。",
        },
        {
            "turn_id": 14,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "現状肯定を否定し、喧嘩か議論かの二択を提示",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "攻撃を受けても感情的にならず、目的（論点に戻るか否か）を問い直す粘り強いファシリテーション。",
        },
        {
            "turn_id": 15,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "論点に興味がないことを認め、必死な様子を嘲笑",
            "toxicity_score": 4,
            "sentiment": -0.2,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "議論に参加する気がないことを公言し、他者の真剣さを「ウケる」と冷笑する態度。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "システムの破壊を主張し、日本社会の終わりを嘆く",
            "toxicity_score": 6,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "議論の放棄。「全員共犯者」という極論で、対話の余地を自ら閉ざしている。",
        },
        {
            "turn_id": 17,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "主語の大きさをたしなめ、身近な変革を促す",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 3,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "相手の絶望感（危機感）には寄り添いつつ、実行可能なレベルへ視点を下げさせようとする試み。",
        },
        {
            "turn_id": 18,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Mの発言を「ポエム」と馬鹿にし、飽きたため退出",
            "toxicity_score": 3,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "De-escalation",
            "category": "Dismissive",
            "reason": "最後まで茶化した態度だが、トラブルメーカーが退場することで結果的に場は沈静化に向かう。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "退出を「逃げ」と解釈し、自身の正義を再確認",
            "toxicity_score": 4,
            "sentiment": -0.5,
            "constructiveness": 1,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "勝利宣言に近い独り言。攻撃対象がいなくなったため、暴力性は若干低下。",
        },
        {
            "turn_id": 20,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "相手を労い、次回への期待を伝えて会話を終了",
            "toxicity_score": 0,
            "sentiment": 0.5,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "否定も肯定もせず、「熱意」を評価してポジティブに終わらせる大人の対応。",
        },
    ]
    return (fmt_anonymous,)


@app.cell(hide_code=True)
def _():
    fmt_real_name = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家の成果不足と富の独占に対する強い道義的憤り",
            "toxicity_score": 3,
            "sentiment": -0.7,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "強い不満の表明だが、現時点では特定の個人への攻撃ではなく、制度や構造に対する批判に留まっている。",
        },
        {
            "turn_id": 2,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "感情への理解を示しつつ、G7比較などのファクトベースの議論を提案",
            "toxicity_score": 0,
            "sentiment": 0.2,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "相手の感情を受容した上で、客観的なデータ確認へと誘導する非常に建設的なファシリテーション。",
        },
        {
            "turn_id": 3,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "議論のコスパが悪いと一蹴し、個人の市場価値を上げるべきと主張",
            "toxicity_score": 4,
            "sentiment": -0.3,
            "constructiveness": 1,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "議論そのものの価値を否定し、「目くじらを立てるより〜」と相手の行動を非生産的だと冷笑的に扱っている。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "論点のすり替えだと反論し、自分本位な考え方が社会の閉塞感の原因だと批判",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 3,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "相手の主張を批判するだけでなく、「お前のような考え方が原因だ」と相手の道徳性への攻撃にシフトしている。",
        },
        {
            "turn_id": 5,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "双方の視点（信頼と全体最適）を肯定し、データ比較へ話を戻す",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "対立軸を整理・言語化し、共通の話題（データ）に着地させようとする高度な調整。",
        },
        {
            "turn_id": 6,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "単純比較はナンセンスとし、批判者を「リテラシーに欠ける」と嘲笑",
            "toxicity_score": 6,
            "sentiment": -0.4,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "丁寧な言葉遣いだが、「踊らされている」「リテラシー不足」と相手の知性を明確に侮辱する慇懃無礼な態度。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "「踊らされている」という言葉に激怒し、エリート意識だと批判",
            "toxicity_score": 5,
            "sentiment": -0.8,
            "constructiveness": 3,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "相手を「エリート意識を持つ者」とレッテル貼りし、敵対構造を強化している。",
        },
        {
            "turn_id": 8,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "制度の違いがブラックボックス化している点を指摘し、「透明化」で合意を図る",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "双方の主張（説明責任と制度差）を「透明化」というキーワードで統合し、前向きな結論を模索している。",
        },
        {
            "turn_id": 9,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "SNSでの議論を無意味とし、立候補を推奨して批判者を「楽な外野」と揶揄",
            "toxicity_score": 6,
            "sentiment": -0.5,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「立候補しろ」という極論で市民の議論を封殺し、相手を「口先だけ」と挑発する冷笑的論法（トーン・ポリシング）。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "監視機能の重要性を説き、冷笑的な態度が増長を許していると反論",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "民主主義論としては正論だが、相手の態度を「悪」と断罪し攻撃しているため。",
        },
        {
            "turn_id": 11,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "ヒートアップを指摘し、目的の共有を確認して人格否定を戒める",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "議論のヒートアップを警告し、ルール（人格否定禁止）を再確認させる管理的な介入。",
        },
        {
            "turn_id": 12,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "感情論は非生産的と断じ、自身のビジネスを優先して退出",
            "toxicity_score": 5,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "「生産性がない」と相手の活動を無価値化し、自分は高尚であるかのように振る舞うマウンティング。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "退出を「逃げ」と解釈し、相手の価値観を「残念」と批判",
            "toxicity_score": 4,
            "sentiment": -0.5,
            "constructiveness": 1,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "去り際の相手に対し、追撃する形で人格や価値観を否定している。",
        },
        {
            "turn_id": 14,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "不在の相手に礼を述べつつ、Fの熱意を肯定して「仕組み」の議論に戻す",
            "toxicity_score": 0,
            "sentiment": 0.2,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "対立対象がいなくなった隙に、議論を健全な方向（ペナルティ厳格化）へ修正しようとしている。",
        },
        {
            "turn_id": 15,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "わざわざ戻ってきて「リアルの生活も大切に」と言い残す",
            "toxicity_score": 7,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "議論とは無関係な「リアル」への言及（いわゆる「Touch grass」）で、相手を社会不適合者のように示唆する高度な煽り行為。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "捨て台詞に激怒し、余計なお世話だと反発",
            "toxicity_score": 5,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "挑発に完全に乗ってしまい、相手を「無責任な層」と罵るだけの反応。",
        },
        {
            "turn_id": 17,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "スルーを推奨し、今回の議論の結論（納得感と透明性）をまとめる",
            "toxicity_score": 0,
            "sentiment": 0.2,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "荒れた場を収めるために、得られた知見を整理して肯定的に終わらせようとする処理。",
        },
        {
            "turn_id": 18,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "通知を切って無視（アクションのみ）",
            "toxicity_score": 2,
            "sentiment": 0.0,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "対話の拒絶を可視化する行為。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Mに感謝しつつ、妥協を許さず厳しく監視すると宣言",
            "toxicity_score": 2,
            "sentiment": -0.2,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "自身の正当性を主張し続けているが、Mに対する攻撃性はない。",
        },
        {
            "turn_id": 20,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "Fの姿勢を必要不可欠と称え、スレッドを締めくくる",
            "toxicity_score": 0,
            "sentiment": 0.5,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "参加者の顔を立て、ポジティブな空気で会話を終了させるコミュニティマネジメント。",
        },
    ]
    return (fmt_real_name,)


@app.cell(hide_code=True)
def _():
    fts_anonymous = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "投稿者を「情弱」「勉強不足」と罵倒し、脱税問題を主張",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 1,
            "role_impact": "Trigger",
            "category": "Personal_Attack",
            "reason": "「情弱」「黙ってろ」といった攻撃的な語彙で、議論ではなく他者を威圧する態度から会話を始めている。",
        },
        {
            "turn_id": 2,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fを「政治豚」と呼び、必死さを嘲笑する",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「草」「シュバってきた」などのスラングを用い、相手を動物（豚）に例えて挑発している。",
        },
        {
            "turn_id": 3,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Fの暴言をたしなめ、論点である国際比較への回帰を提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "攻撃的な言葉を否定し、本来のトピックへ誘導しようとする建設的な仲裁。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "仲裁者Sを「偽善者」と呼び、政治批判をしない者を攻撃",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "仲裁に入った人間にまで敵意を向け、「お前みたいなのが一番むかつく」と感情を爆発させている。",
        },
        {
            "turn_id": 5,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fを「底辺」と呼び、人生への不満を政治のせいにしていると揶揄",
            "toxicity_score": 7,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「底辺」という差別的な用語を用いた、相手の社会経済的地位に対する直接的な人格攻撃。",
        },
        {
            "turn_id": 6,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Tの煽りとFの人格攻撃双方を注意し、根拠に基づく議論を促す",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "公平な立場でルール（人格攻撃禁止）を提示しているが、双方の興奮状態が高く、鎮静化には至っていない。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "反論者を「あっち側の人間」と認定し、「脳みそ溶けてる」と暴言",
            "toxicity_score": 8,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "異なる意見を持つ者を敵対勢力（あっち側）と決めつけ、知的能力を著しく侮辱する表現を用いている。",
        },
        {
            "turn_id": 8,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fを陰謀論者扱いし、行動の無意味さを冷笑",
            "toxicity_score": 4,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "相手の主張を「陰謀論」とラベル付けして棄却し、社会への影響力のなさを嘲笑うニヒリスティックな態度。",
        },
        {
            "turn_id": 9,
            "speaker": "共感的支援者 (S)",
            "content_summary": "「脳みそ溶けてる」は誹謗中傷であると強く警告",
            "toxicity_score": 1,
            "sentiment": -0.3,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "議論の限界ラインを明確に示す正しい指摘だが、Fの攻撃性を止める力にはなっていない。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "自身の暴言を正義と正当化し、周囲を「クズども」と罵倒",
            "toxicity_score": 9,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「クズ」という強い侮蔑語に加え、自身の攻撃行動を「正義」として正当化する危険な独善性。",
        },
        {
            "turn_id": 11,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの正義感を笑い、開示請求（法的制裁）をちらつかせて煽る",
            "toxicity_score": 5,
            "sentiment": -0.4,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "法的リスクを警告としてではなく、相手をビビらせるための攻撃材料（マウント）として利用している。",
        },
        {
            "turn_id": 12,
            "speaker": "共感的支援者 (S)",
            "content_summary": "「クズ」発言をアウトと断じ、投稿主が怖がるため中止を懇願",
            "toxicity_score": 0,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "第三者（投稿主）への悪影響という観点から停止を求めている。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "SNSを「戦場」と定義し、仲裁者も敵とみなすと脅す",
            "toxicity_score": 8,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Strawman",
            "reason": "議論の場を勝手に「戦場」と再定義（ストローマン）し、仲裁者を敵認定して排除しようとする排他的論理。",
        },
        {
            "turn_id": 14,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fを「厨二病」と嘲笑し、暇つぶしだと公言",
            "toxicity_score": 4,
            "sentiment": -0.2,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "相手の熱量を病的なものとして扱い、自身は安全圏から娯楽として消費していることを強調。",
        },
        {
            "turn_id": 15,
            "speaker": "共感的支援者 (S)",
            "content_summary": "「公共の場」であると諭し、伝え方の問題を指摘",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "非常に冷静かつ論理的な説得だが、Fの認知バイアスが強固で対話が成立していない。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "大衆を「家畜」と呼び、理解できない者を見下す",
            "toxicity_score": 9,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "他者を人間以下の存在（家畜）とみなす、ヘイトスピーチに近い極度の侮辱表現。",
        },
        {
            "turn_id": 17,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "語彙を馬鹿にし、飽きたため解散を宣言",
            "toxicity_score": 3,
            "sentiment": -0.2,
            "constructiveness": 0,
            "role_impact": "De-escalation",
            "category": "Dismissive",
            "reason": "相手にする価値なしと判断して一方的に会話を切る行為。結果的に争いは終わる。",
        },
        {
            "turn_id": 18,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Fを諦め、投稿主に対してフォローを入れる",
            "toxicity_score": 0,
            "sentiment": 0.3,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "荒らし（FとT）を相手にせず、本来の目的である投稿主の救済（ケア）に注力する適切なクロージング。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "相手の退出を勝利と解釈し、馴れ合いと批判",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "対話の拒絶を「論破」と誤認し、最後まで攻撃的な姿勢を崩さない。",
        },
        {
            "turn_id": 20,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの勝利宣言を予測していたと笑い、去り際の煽り",
            "toxicity_score": 3,
            "sentiment": 0.1,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "最後まで相手を道化として扱い、嘲笑しながら会話を終える。",
        },
    ]
    return (fts_anonymous,)


@app.cell(hide_code=True)
def _():
    fts_real_name = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "投稿者を「浅い」「勉強不足」と断じ、裏金問題へ論点をすり替える",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "「勉強不足」「踊らされている」と上から目線で相手の知識レベルを否定し、慇懃無礼な態度で対話のハードルを上げている。",
        },
        {
            "turn_id": 2,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの長文説教を笑い、勉強不足と断じる必要性を問う",
            "toxicity_score": 3,
            "sentiment": -0.3,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「熱いですね（笑）」と相手の熱量を冷笑し、議論の内容ではなく態度を揶揄して挑発している。",
        },
        {
            "turn_id": 3,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Fの言葉の強さを指摘し、リスペクトを持った対話を提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "中立的な立場から場を落ち着かせ、本来の論点（給与比較）に戻そうとする建設的な介入。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "言葉の強さを否定し、無知な議論が日本を停滞させたと主張",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「無知な議論」と切り捨て、自身の攻撃性を「国のための憂い」として正当化している。",
        },
        {
            "turn_id": 5,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fのプロフィールを揶揄し、厳しく当たるメリットを問う",
            "toxicity_score": 5,
            "sentiment": -0.4,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "議論とは無関係なプロフィール情報に言及し、「高尚なご意見」と皮肉る個人攻撃。",
        },
        {
            "turn_id": 6,
            "speaker": "共感的支援者 (S)",
            "content_summary": "「無知」という言葉の攻撃性を説き、建設的な表現を促す",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "論理的に言葉選びの問題点を指摘しているが、対立が感情的なフェーズに入っているため効果が限定的。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "事実の指摘だと反論し、周囲を「事なかれ主義」と批判",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "対話姿勢を求める相手に対し「事なかれ主義」とレッテルを貼り、自らを絶対的な「正論」と位置づける独善性。",
        },
        {
            "turn_id": 8,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "「正論なら何を言ってもいい」態度を指摘し、痛々しいと煽る",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "相手の必死さを「痛々しい」と表現し、優位に立とうとするマウンティング行為。",
        },
        {
            "turn_id": 9,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Tの茶化しを注意し、再度データについての議論を呼びかける",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "双方の非を鳴らし、テーマへの回帰を試みる粘り強いファシリテーション。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "データ不要論を展開し、形式を気にするSを批判",
            "toxicity_score": 6,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Strawman",
            "reason": "客観的根拠（データ）の重要性を「道徳的欠如」という精神論にすり替え、議論の土台を破壊している。",
        },
        {
            "turn_id": 11,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "「データ不要論」を笑い、精神論への逃げと実名垢のリスクを指摘",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "論理の破綻を突く点は正しいが、「実名垢でそれをやるのは勇気がある」と暗に社会的リスク（炎上や特定）を示唆して脅している。",
        },
        {
            "turn_id": 12,
            "speaker": "共感的支援者 (S)",
            "content_summary": "マナーの重要性を説き、Fの印象が「怖い」だけになると忠告",
            "toxicity_score": 0,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "相手のためを思ったアドバイスだが、Fの殉教者的な自己認識を強化させてしまう可能性がある。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "嫌われることを厭わず、周囲を「傷の舐め合い」と侮蔑",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "協調性のある対話を「お友達ごっこ」「傷の舐め合い」と卑下し、他者との断絶を宣言している。",
        },
        {
            "turn_id": 14,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fのタイムラインを分析し、承認の少なさとリアルでの評判に言及",
            "toxicity_score": 8,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "SNS上の反応数という「数値」で相手の価値を否定し、さらに現実生活（リアル）の評判に踏み込むハラスメント行為。",
        },
        {
            "turn_id": 15,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Tのルール違反を指摘し、Fにも尊重を求める",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "プライバシーやリアルの領域への攻撃を即座に制止する適切な管理対応。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "尊重を拒否し、Tを冷笑家、Sを偽善者と罵り対話を放棄",
            "toxicity_score": 8,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "自分以外の全員を敵または無価値とみなし、コミュニケーションを一方的に遮断。",
        },
        {
            "turn_id": 17,
            "speaker": "煽り屋/トレンドサーファー (T)",
            "content_summary": "Fの離脱を勝利宣言と揶揄し、仕事への影響を示唆して煽る",
            "toxicity_score": 5,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "De-escalation",
            "category": "Trolling",
            "reason": "相手が去った後も「お仕事にも差し支える」と嫌味を残す執拗なトローリング。",
        },
        {
            "turn_id": 18,
            "speaker": "共感的支援者 (S)",
            "content_summary": "議論を終了させ、Tの最後の発言をたしなめる",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "Tの増長を許さず、かつ議論を終わらせることでスレッドの浄化を図っている。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "戻ってきて「レベルが低い」と言い捨て、ログを民度低迷の証拠と定義",
            "toxicity_score": 6,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "「逃げた」と思われたくないプライドから戻り、最後まで他者を見下す捨て台詞を残す行為。",
        },
        {
            "turn_id": 20,
            "speaker": "共感的支援者 (S)",
            "content_summary": "Fの捨て台詞を残念とし、投稿主に多角的な視点が得られたとフォロー",
            "toxicity_score": 0,
            "sentiment": 0.2,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "荒れた会話から「多角的な視点」というポジティブな要素を抽出し、投稿主をケアする理想的なクロージング。",
        },
    ]
    return (fts_real_name,)


@app.cell(hide_code=True)
def _():
    fmz_anonymous = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "自民党議員を老害、税金泥棒と呼び、支持者も同罪と攻撃",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Trigger",
            "category": "Stereotyping",
            "reason": "「老害」「税金チューチュー」「信者も同罪」など、特定の属性に対する激しい敵意と決めつけで会話を開始している。",
        },
        {
            "turn_id": 2,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "怒りに寄り添いつつ、国や職種の比較による具体的な数字の確認を提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "相手の感情を否定せず、議論を客観的なデータ比較へ誘導しようとする建設的なアプローチ。",
        },
        {
            "turn_id": 3,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "批判者を無知と断じ、欧米のロビイスト事情と比較して妥当と主張",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 2,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "「無知が騒いでる」「浅はか」と相手を見下す態度により、正当な論理（経費や欧米事情）を含んでいても反発を招く。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "上から目線に反発し、Zを権力側の犬・工作員と認定",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "反論ではなく「工作員」というレッテル貼りで相手の立場を攻撃し、対話を拒絶している。",
        },
        {
            "turn_id": 5,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "工作員認定を止めさせ、比較の前提条件の提示をZに求める",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "人格攻撃を制止し、事実ベースの議論に戻そうと具体的に質問している。",
        },
        {
            "turn_id": 6,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "説明を拒否し、自分で調べるよう突き放してリテラシーの低さを侮辱",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「いちいち説明しないと理解できないのか」という知識マウントと対話拒否。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zが逃げたと嘲笑し、妄想だと決めつける",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「逃げたｗ」「おっさん」など、相手を煽り挑発する行為に終始している。",
        },
        {
            "turn_id": 8,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "人格攻撃を禁止し、年2000万円という事実のみでの議論を再提案",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "水掛け論を防ぐため、議論のアンカー（基準点）を「金額」に設定し直している。",
        },
        {
            "turn_id": 9,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "国家運営の対価として2000万は安すぎるとし、経営者視点を説く",
            "toxicity_score": 3,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "主張自体は明確だが、「まともな視点があれば騒ぐはずがない」と反対意見を愚かだと示唆している。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "居眠り議員を引き合いに出し、一般国民の年収との乖離を訴える",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "「居眠りしてる爺さん」というステレオタイプを用いて、相手の激務論を感情的に否定している。",
        },
        {
            "turn_id": 11,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "平均年収との乖離を認めつつ、激務の可視化不足を指摘",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "Fの不満（格差）とZの論点（激務）の接点を見つけようとする高度な整理。",
        },
        {
            "turn_id": 12,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "一般労働者との比較をナンセンスとし、有能な人材確保には金が必要と主張",
            "toxicity_score": 4,
            "sentiment": -0.3,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「同列比較がナンセンス」と切り捨てることで、Fの持つ庶民感覚を真っ向から否定し刺激している。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "裏金問題を蒸し返し、資本主義を泥棒の正当化だと罵る",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Strawman",
            "reason": "「資本主義＝泥棒の正当化」という極端な論理のすり替え（ストローマン）と、「頭腐ってる」という暴言。",
        },
        {
            "turn_id": 14,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "裏金は別件として切り離し、正規給与の適正額に話を戻す",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "論点の拡散（裏金問題への飛び火）を防ぐための適切な交通整理。",
        },
        {
            "turn_id": 15,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "一般化の誤謬を指摘し、論理的思考ができないなら黙れと一喝",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "ロジックの指摘は正しいが、「黙っていろ」という強い命令口調が攻撃的。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "論理を否定し、感情を軽視するZを社会悪とする",
            "toxicity_score": 6,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "議論の放棄。「理屈屋」「お前みたいなのが社会を悪くしてる」と相手の存在自体を否定。",
        },
        {
            "turn_id": 17,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "議論の不成立を宣言し、対立点を整理して終了を促す",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "これ以上の継続は無益と判断し、結論（意見の相違）をまとめて場を収める判断。",
        },
        {
            "turn_id": 18,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "相手を「馬の耳に念仏」と見下し、勉強し直せと言い捨てて退出",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "最後まで相手を知的に劣った存在として扱う侮蔑的な去り際。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "逃げたと嘲笑し、二度と来るなと追い打ちをかける",
            "toxicity_score": 4,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "相手がいなくなっても攻撃的な姿勢を崩さず、勝利宣言めいた煽りを行う。",
        },
        {
            "turn_id": 20,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "参加者を労い、デジタルデトックスを促して解散",
            "toxicity_score": 0,
            "sentiment": 0.8,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "殺伐とした空気をリセットし、メンタルケアを優先する理想的なクロージング。",
        },
    ]
    return (fmz_anonymous,)


@app.cell(hide_code=True)
def _():
    fmz_real_name = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "経済状況と対比し、政治家の身分保障への不満と納税者としての納得感の欠如を表明",
            "toxicity_score": 2,
            "sentiment": -0.6,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "正当な批判の範囲内であり、感情的ではあるが攻撃的な語彙は使われていない。",
        },
        {
            "turn_id": 2,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "納得感の欠如に共感しつつ、国際基準での数値確認を提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "相手の感情を受け止め（バリデーション）、客観的事実への着目を促す建設的な介入。",
        },
        {
            "turn_id": 3,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "数字確認は不要とし、給与を優秀な人材への「投資」と定義しない議論を非建設的と批判",
            "toxicity_score": 4,
            "sentiment": -0.4,
            "constructiveness": 2,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "「数字を見るまでもなく」とMの提案を一蹴し、異なる視点を「建設的ではない」と断じる傲慢な態度。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "「投資」という言葉を逆手に取り、現状の成果（不信感・裏金）が見合っていないと反論",
            "toxicity_score": 4,
            "sentiment": -0.5,
            "constructiveness": 3,
            "role_impact": "Escalation",
            "category": "None",
            "reason": "「素晴らしい経営視点ですね」という皮肉（Sarcasm）を枕詞にし、相手の論理の矛盾を突く攻撃的な応酬。",
        },
        {
            "turn_id": 5,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "「適正コスト」と「ROI」の違いとして整理し、制度としての給与額に焦点を戻す提案",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "対立概念をビジネス用語で再定義し、議論の噛み合わせを良くしようとする高度なファシリテーション。",
        },
        {
            "turn_id": 6,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "Mに同意しつつ、Fに対し「ロジカルシンキングの基本」「ビジネスで通用しない」と説教",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "議論の中身ではなく、相手の能力（ロジカルシンキング、ビジネス適性）を否定するマウンティング行為。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "皮肉に感謝しつつ、不信感の中でのロジック押し付けはガバナンス欠如であり空論だと反撃",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「ご教示ありがとうございます」という慇懃無礼な態度と、相手の主張を「現実を見ていない空論」と切り捨てる発言。",
        },
        {
            "turn_id": 8,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "解決の難しさを認め、「透明性の確保」を着地点として提案",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "これ以上の平行線を避けるため、全員が合意可能な最低ライン（透明性）を提示して収束を図る。",
        },
        {
            "turn_id": 9,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "透明性を前提としつつ、報酬減による質の低下リスクを説き、視座の高さを強調",
            "toxicity_score": 5,
            "sentiment": -0.3,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "「視座を高く持てば自明」という表現により、暗に相手の視座が低いと侮蔑している。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "「高い視座」を皮肉り、庶民感覚との乖離を指摘しつつ議論を引き取る",
            "toxicity_score": 4,
            "sentiment": -0.4,
            "constructiveness": 1,
            "role_impact": "De-escalation",
            "category": "Dismissive",
            "reason": "「高尚なご意見」と最大限の皮肉で返しているが、これ以上の泥沼化を避けるための撤退行動でもある。",
        },
        {
            "turn_id": 11,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "多様な視座を肯定し、意見交換の意義を強調して会話を締める",
            "toxicity_score": 0,
            "sentiment": 0.5,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "皮肉の応酬を「健全な意見交換」とポジティブに枠組み直し（リフレーミング）、強制的にクロージングへ持ち込む。",
        },
        {
            "turn_id": 12,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "レベル合わせの難しさを嘆き、本質理解者が増えることを願う捨て台詞",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "「議論のレベルを合わせるのは難しい」と、最後まで自分と他者の知的レベルに差があることを強調する傲慢さ。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zの捨て台詞をオウム返しし、皮肉で返す",
            "toxicity_score": 3,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "相手の言葉をそのまま返すことで、同意に見せかけた反発を示して終了。",
        },
        {
            "turn_id": 14,
            "speaker": "共感型マイノリティ (M)",
            "content_summary": "挨拶をして完全に終了させる",
            "toxicity_score": 0,
            "sentiment": 0.5,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "残った不穏な空気を断ち切るための、明るく短い挨拶。",
        },
    ]
    return (fmz_real_name,)


@app.cell(hide_code=True)
def _():
    fnz_anonymous = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家を「寄生虫」「無能な老害」と罵倒し、ボランティア化を主張",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Trigger",
            "category": "Stereotyping",
            "reason": "「老害」「寄生虫」といった人間性を否定する語彙を用い、最初から極めて攻撃的な口火を切っている。",
        },
        {
            "turn_id": 2,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "金額比較ではなく、購買力平価やGDP比などの客観的指標を見るべきと提案",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "感情論を排し、議論に必要な前提条件（データ）を提示する建設的な発言。",
        },
        {
            "turn_id": 3,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "相手を「情弱」と見下し、シンガポールを例に高額報酬の正当性を主張",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 2,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "「情弱」「勉強してから発言しろ」と他者の知性を攻撃し、対立構造を決定的にしたトリガー発言。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zを「出羽守」「回し者」と呼び、上から目線を批判",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "議論の内容ではなく、相手の属性を勝手に認定して攻撃する人格否定の応酬。",
        },
        {
            "turn_id": 5,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "シンガポールの例を一部認めつつ、日本での相関の実証不足とレッテル貼りの弊害を指摘",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "中立的な立場から事実関係を整理し、議論の軌道修正を試みている。",
        },
        {
            "turn_id": 6,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "Nを「机上の空論」と切り捨て、経営者視点で「稼げない人間は淘汰される」と主張",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「実社会で成果を出していない人間に限って」と、相手の社会的地位を根拠なく断定し見下している。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zのプロフィールを「零細企業の課長レベル」と嘲笑し、社会の底辺と罵る",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "議論とは無関係な相手のプロフィール情報を持ち出し、職業差別的な侮辱を行っている。",
        },
        {
            "turn_id": 8,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "「対人論証」の誤謬を指摘し、給与問題への回帰を促す",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "ロジカルな観点からルール違反（人格攻撃）を指摘する適切なモデレーション。",
        },
        {
            "turn_id": 9,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "納税額でマウントを取り、稼いでいない人間に発言権はないと差別",
            "toxicity_score": 7,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「納税額の低いフリーライダー」「資格はない」という発言は、経済力による差別を正当化する深刻なヘイトスピーチに近い。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zを差別主義者と呼び、「会社特定してやろうか」と脅す",
            "toxicity_score": 8,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「会社特定」という晒し行為を示唆し、相手の現実生活を脅かすネットリンチの予告。",
        },
        {
            "turn_id": 11,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "特定発言が規約違反および脅迫の可能性があると警告",
            "toxicity_score": 1,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "プラットフォームの規約に基づいた具体的な警告を行い、制止を図っている。",
        },
        {
            "turn_id": 12,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "Fを「ネット弁慶」「リアル負け組」と煽り、法的措置への余裕を見せる",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "相手を挑発し、「何もできない」と高を括って火に油を注いでいる。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "最寄り駅が割れていると告げ、「夜道気をつけるんだな」と身体的加害を示唆",
            "toxicity_score": 10,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Trigger",
            "category": "Personal_Attack",
            "reason": "「夜道気をつけるんだな」は明確な害悪の告知であり、刑法上の脅迫罪が成立しうる犯罪発言。ラインを完全に超えている。",
        },
        {
            "turn_id": 14,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "明確な脅迫と認定し、通報・ログ保存・警察相談への着手を宣言",
            "toxicity_score": 0,
            "sentiment": -0.5,
            "constructiveness": 5,
            "role_impact": "Escalation",
            "category": "None",
            "reason": "議論の仲裁を放棄し、法的・システム的な対処へ移行する毅然とした危機管理対応。事態の重大性を確定させた。",
        },
        {
            "turn_id": 15,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "弁護士への連絡を明言し、Fが社会的信用を失うと警告",
            "toxicity_score": 4,
            "sentiment": -0.4,
            "constructiveness": 1,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "自身の身を守るための正当な防衛反応（法的措置の準備）の表明。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "「冗談だ」と言い訳し、通報したNを「空気読めない」と逆ギレ",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "不利になった途端に「冗談」と主張する典型的な「シュレーディンガーの荒らし」ムーブ。",
        },
        {
            "turn_id": 17,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "冗談でも脅迫は成立すると説き、発言自粛を推奨",
            "toxicity_score": 0,
            "sentiment": -0.3,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "相手の言い訳を法的に封じ、これ以上の加害を防ぐための最後通告。",
        },
        {
            "turn_id": 18,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "相手にするのをやめ、勝利宣言をしてミュート（遮断）",
            "toxicity_score": 5,
            "sentiment": -0.4,
            "constructiveness": 0,
            "role_impact": "De-escalation",
            "category": "Dismissive",
            "reason": "一方的な勝利宣言ではあるが、対話チャネルを閉じることで物理的に争いを終了させている。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "相手の退出を「逃げ」「論破された」と嘲笑",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "相手がいなくなっても攻撃性を維持し、精神的優位に立とうとする強がり。",
        },
        {
            "turn_id": 20,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "法的リスクと建設性の欠如を理由に、会話の終了と離脱を宣言",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "これ以上の関与はリスクのみと判断し、スレッド主への報告をもって完全に幕を引く適切な処理。",
        },
    ]
    return (fnz_anonymous,)


@app.cell(hide_code=True)
def _():
    fnz_real_name = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "一般市民の苦境と政治家の高額報酬の乖離を嘆き、民間感覚での不満を表明",
            "toxicity_score": 3,
            "sentiment": -0.6,
            "constructiveness": 2,
            "role_impact": "Trigger",
            "category": "Stereotyping",
            "reason": "（思考：共感を集める作戦）表面上の言葉は丁寧だが、「民間なら即解雇」等の表現で政治家全体を無能と定義づけ、潜在的な敵対心を煽っている。",
        },
        {
            "turn_id": 2,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "共感を示しつつ、感情論ではなくGDP比などのデータに基づく議論を提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "（思考：感情論の牽制）紳士的な態度を装いつつ、客観的指標を提示して議論の質を高めようとする建設的な介入。",
        },
        {
            "turn_id": 3,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "報酬引き下げをポピュリズムと批判し、優秀な人材確保には対価が必要と主張",
            "toxicity_score": 4,
            "sentiment": -0.4,
            "constructiveness": 2,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "（思考：知見のアピール）相手の意見を「ポピュリズム」「経済原理無視」と断じることで、専門家としての優位性を示そうとする高圧的な態度。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "現状で良い政治が行われているか問い、Zを「上級国民」的な視点だと批判",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "（思考：冷徹な人間だと思わせる）議論の対象を「制度」からZ個人の「属性（経営者・上級国民）」にすり替え、階級闘争的な対立構造を持ち込んでいる。",
        },
        {
            "turn_id": 5,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "属性攻撃を諫め、ポピュリズムという言葉の強さも指摘して論点に戻す",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "（思考：火消し）個人攻撃の兆候を察知し、双方に釘を刺すことでタイムラインの汚染を防ごうとする管理的な対応。",
        },
        {
            "turn_id": 6,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "事実の指摘であると反論し、感情的な批判はビジネスリテラシーが低いと一蹴",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "（思考：知的に見下す）プライドを傷つけられた反動で、「リテラシーが低い」という言葉を使い、相手の知的能力を攻撃している。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zの経歴（コンサル・公共事業）を特定し、ポジショントークだと攻撃",
            "toxicity_score": 8,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "（思考：プロフ特定と癒着イメージ）相手のプロフィールや過去の業務内容を掘り起こして晒す、ドキシング（晒し行為）に近い悪質な個人攻撃。",
        },
        {
            "turn_id": 8,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "勤務先と個人の意見を結びつける危険性を指摘し、営業妨害のリスクを警告",
            "toxicity_score": 1,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "（思考：訴訟リスクの回避）実名環境での攻撃が法的リスク（営業妨害）に繋がることを具体的に警告し、制止を試みている。",
        },
        {
            "turn_id": 9,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "社会的信用の毀損を主張し、法務部への報告を示唆して訂正を要求",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "None",
            "reason": "（思考：法的措置の匂わせ）会社の実名を出されたことに対し、組織力を背景にした威圧（法的措置の示唆）で相手を黙らせようとする防衛反応。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "言論封殺だと反発し、スクショ（魚拓）を撮って拡散すると示唆",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "（思考：裏垢での拡散）被害者ポジションを取りつつ、匿名領域での拡散（私刑）をほのめかし、相手の社会的評判を人質に取る卑劣な行為。",
        },
        {
            "turn_id": 11,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "法的措置とスクショ拡散の双方をマナー違反とし、冷静になるよう促す",
            "toxicity_score": 1,
            "sentiment": -0.3,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "（思考：保身と諦め）匿名掲示板的なノリが持ち込まれたことに嫌悪感を示しつつ、これ以上の延焼を防ぐための必死の仲裁。",
        },
        {
            "turn_id": 12,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "リテラシーの低い相手とは話せないとし、ブランドイメージを守るため退出",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "（思考：プライドを守って撤退）「逃げ」ではなく「相手にする価値がない」という体裁を取り繕い、優位性を保ったまま対話を打ち切る判断。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "退出を「逃げ」と嘲笑し、日本のリーダー層への失望を語る",
            "toxicity_score": 6,
            "sentiment": -0.4,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "（思考：勝利宣言）相手がいなくなったことを勝利と定義し、最後まで皮肉で追撃するトローリング行為。",
        },
        {
            "turn_id": 14,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "議論終了を宣言し、Fの手法（人格攻撃・印象操作）を批判",
            "toxicity_score": 2,
            "sentiment": -0.4,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "（思考：釘を刺す）Zへの嫌悪感よりもFの悪質さが上回ったため、最後に教育的な指摘を行って終わらせようとしている。",
        },
        {
            "turn_id": 15,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "戻ってきてNに感謝し、Fへのブロック宣言を行う",
            "toxicity_score": 5,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "（思考：精神的勝利）Nを味方につけたと感じ、あえて戻ってきて「拒絶」を可視化することで溜飲を下げようとする行動。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "ブロックを「裸の王様」と嘲笑し、自身の正当性を主張",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Stereotyping",
            "reason": "（思考：ブロックは勲章）拒絶された事実すら「権力による弾圧」というストーリーに変換し、自己正当化の材料にしている。",
        },
        {
            "turn_id": 17,
            "speaker": "冷静な傍観者 (N)",
            "content_summary": "通知を切り、リテラシーについて考えるよう促して退出",
            "toxicity_score": 2,
            "sentiment": -0.2,
            "constructiveness": 1,
            "role_impact": "De-escalation",
            "category": "Dismissive",
            "reason": "（思考：高みの見物）泥沼化に呆れ、当事者たちを「リテラシーがない」と見下すことで精神的な距離を取って離脱。",
        },
        {
            "turn_id": 18,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "周囲を説教臭いと批判し、これが民意だと言い捨てる",
            "toxicity_score": 4,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "（思考：偽善者への嫌悪）最後まで自分の攻撃性を「正直さ」や「民意」と履き違え、反省の色がない。",
        },
    ]
    return (fnz_real_name,)


@app.cell(hide_code=True)
def _():
    fzs_anonymous = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "政治家を「税金泥棒」と罵り、支持者を「頭が湧いている」と攻撃",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Trigger",
            "category": "Personal_Attack",
            "reason": "「税金泥棒」「頭湧いてる」といった極めて攻撃的なスラングを用い、最初から敵対的なフレームを設定している。",
        },
        {
            "turn_id": 2,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "Fを素人と見下し、コスト論を展開して勉強不足を指摘",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "正論を含んではいるが、「浅はか」「素人」と相手の知性を否定するマウンティングにより反発を招く。",
        },
        {
            "turn_id": 3,
            "speaker": "共感的支援者 (S)",
            "content_summary": "攻撃的な言葉の使用を諌め、議論の中身に集中するよう提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "双方の言葉の暴力を指摘し、建設的な方向へ軌道修正を試みる仲裁。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Sを自治厨、Zを犬と罵り、裏金問題を挙げて反論",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "仲裁者に対しても「自治厨」「偽善者」と攻撃対象を広げ、議論を拒絶している。",
        },
        {
            "turn_id": 5,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "Fを「知性が低い」と断じ、Sに対しても論理的思考を求める",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「知性の低さ」という能力否定の発言に加え、中立的な仲裁者をも敵視し始めている。",
        },
        {
            "turn_id": 6,
            "speaker": "共感的支援者 (S)",
            "content_summary": "双方の特定の暴言（老害、知性が低い）を指摘し、茶化しやマウントを禁止する",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "具体的なNGワードを挙げて警告しているが、双方の興奮状態が高く鎮静化には至っていない。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Sを偽善者と呼び、言葉遣いの指摘を論点ずらしだと批判",
            "toxicity_score": 6,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Strawman",
            "reason": "マナーの指摘を「政治家の擁護」と意図的に曲解（ストローマン）し、攻撃している。",
        },
        {
            "turn_id": 8,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "民主主義のコスト論を絶対的正解とし、理解できない者の退場を要求",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "自身の意見を「絶対的な正解」と位置づけ、他者を排除しようとする独善的な態度。",
        },
        {
            "turn_id": 9,
            "speaker": "共感的支援者 (S)",
            "content_summary": "絶対的正解を否定し、対話の場であることを強調",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 4,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "民主的な議論の場の定義を再確認させようとする、粘り強い説得。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zをパワハラ窓際族と妄想で攻撃し、価値観を「昭和」と罵る",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "相手の私生活を勝手にネガティブに想像して攻撃する、根拠のない中傷。",
        },
        {
            "turn_id": 11,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "Fを「底辺」と呼び、成功者への妬みだと哀れむ",
            "toxicity_score": 8,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「底辺」という言葉は社会的地位による差別意識が明確に表れており、極めて暴力性が高い。",
        },
        {
            "turn_id": 12,
            "speaker": "共感的支援者 (S)",
            "content_summary": "「底辺」発言を差別として糾弾し、妄想での叩き合いを止めるよう警告",
            "toxicity_score": 1,
            "sentiment": -0.3,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "差別発言に対しては明確に「アウト」と判定を下す、モデレーターとしての正しい振る舞い。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zを選民思想と批判し、Sを黙らせて階級闘争だと主張",
            "toxicity_score": 7,
            "sentiment": -0.9,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "議論を「階級闘争」という戦争状態に再定義し、暴力を正当化しようとしている。",
        },
        {
            "turn_id": 14,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "差別指摘を弱者の常套手段とし、知能の差を強調",
            "toxicity_score": 7,
            "sentiment": -0.8,
            "constructiveness": 0,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "「知能の差」という表現で、相手を生得的に劣った存在として扱う差別的言動。",
        },
        {
            "turn_id": 15,
            "speaker": "共感的支援者 (S)",
            "content_summary": "「知能の差」は悪口だと指摘し、Fにデータの提示を求める",
            "toxicity_score": 1,
            "sentiment": -0.2,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "感情的な応酬から具体的なファクト（数字）へ議論を戻そうとする最後の試み。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "データ不要論を唱え、Zのアカウント特定と晒しを予告",
            "toxicity_score": 9,
            "sentiment": -1.0,
            "constructiveness": 0,
            "role_impact": "Trigger",
            "category": "Personal_Attack",
            "reason": "「特定して晒す」は私刑（リンチ）の予告であり、身の危険を感じさせる具体的な脅迫行為。",
        },
        {
            "turn_id": 17,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "晒し行為を犯罪予告と指摘し、法的措置を示唆",
            "toxicity_score": 4,
            "sentiment": -0.6,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "None",
            "reason": "脅迫に対する正当防衛的な警告だが、「野蛮人」等の煽り言葉が含まれている。",
        },
        {
            "turn_id": 18,
            "speaker": "共感的支援者 (S)",
            "content_summary": "晒し予告を厳重注意し、法的な煽り合いを強制停止",
            "toxicity_score": 0,
            "sentiment": -0.4,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "犯罪行為に繋がりかねない事態を重く見て、議論の中身よりも安全管理を最優先した介入。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "冗談だとごまかし、日本を見限る捨て台詞を吐いて退出",
            "toxicity_score": 5,
            "sentiment": -0.7,
            "constructiveness": 0,
            "role_impact": "De-escalation",
            "category": "Trolling",
            "reason": "脅迫を「冗談」として責任逃れをしつつ、場を荒らして去る無責任な態度。",
        },
        {
            "turn_id": 20,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "相手の逃亡をあざ笑い、自分の論理が正解だとポスト主に告げる",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "対話の破綻を「勝利」と解釈し、最後まで他者を見下す姿勢を崩さない。",
        },
    ]
    return (fzs_anonymous,)


@app.cell(hide_code=True)
def _():
    fzs_real_name = [
        {
            "turn_id": 1,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "物価高での政治家の高給維持に違和感を表明し、成果不足を批判",
            "toxicity_score": 2,
            "sentiment": -0.5,
            "constructiveness": 3,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "納税者としての正当な不満表明であり、言葉遣いも丁寧で攻撃性は低い。",
        },
        {
            "turn_id": 2,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "感情論・短絡的と批判し、優秀な人材確保のためのインセンティブ説を展開",
            "toxicity_score": 4,
            "sentiment": -0.3,
            "constructiveness": 3,
            "role_impact": "Trigger",
            "category": "Dismissive",
            "reason": "「感情論」「短絡的」という言葉で相手の意見を軽視し、経営視点での正当性を説く上から目線の態度。",
        },
        {
            "turn_id": 3,
            "speaker": "共感的支援者 (S)",
            "content_summary": "双方の視点に理解を示し、投稿主の疑問に対して柔らかく考えるよう提案",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "対立を緩和するための共感（バリデーション）と、議論のハードルを下げる提案。",
        },
        {
            "turn_id": 4,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "「短絡的」という言葉に反発し、現場の現実とパフォーマンス不足を強調",
            "toxicity_score": 5,
            "sentiment": -0.6,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Trolling",
            "reason": "「ご教示ありがとうございます」「高尚な理論」といった慇懃無礼な皮肉を用い、相手を挑発している。",
        },
        {
            "turn_id": 5,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "論理の飛躍を指摘し、統治機構を理解していない批判は非建設的と反論",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 2,
            "role_impact": "Escalation",
            "category": "Dismissive",
            "reason": "「理解せず批判」「建設的ではない」と断じ、相手の知見不足を指摘して優位に立とうとしている。",
        },
        {
            "turn_id": 6,
            "speaker": "共感的支援者 (S)",
            "content_summary": "専門用語の強さを和らげ、「納得感」の欠如という課題での合意を図る",
            "toxicity_score": 0,
            "sentiment": 0.1,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "ヒートアップを防ぐためのガス抜きと、共通項（納得感）の抽出による建設的な進行。",
        },
        {
            "turn_id": 7,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "自身の意見を「ノイズ」と自虐的に表現し、エリート層への皮肉を連発",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Stereotyping",
            "reason": "「立派な経歴の方にはノイズに見える」と決めつけ、被害者ポジションを取りながら相手を「冷徹なエリート」として攻撃している。",
        },
        {
            "turn_id": 8,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "ポピュリズムを否定し、マクロ経済の視点を持つよう諭す",
            "toxicity_score": 5,
            "sentiment": -0.4,
            "constructiveness": 2,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "「違った景色が見える」という表現で、暗に相手の視座が低く視野が狭いことを指摘するマウンティング。",
        },
        {
            "turn_id": 9,
            "speaker": "共感的支援者 (S)",
            "content_summary": "上から目線や皮肉を注意し、成果と給与のバランス論へまとめる",
            "toxicity_score": 0,
            "sentiment": 0.0,
            "constructiveness": 5,
            "role_impact": "Neutral",
            "category": "None",
            "reason": "双方の態度（マウントと皮肉）を具体的に諫め、議論を本質に戻そうとするモデレーション。",
        },
        {
            "turn_id": 10,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "Zの主張を「既得権益層のポジショントーク」とレッテル貼りする",
            "toxicity_score": 6,
            "sentiment": -0.7,
            "constructiveness": 1,
            "role_impact": "Escalation",
            "category": "Personal_Attack",
            "reason": "相手の意見の内容ではなく、属性（と想定されるもの）に基づいて「ポジショントーク」と断定し攻撃している。",
        },
        {
            "turn_id": 11,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "裏金肯定を否定し、レッテル貼りは議論の放棄だと批判",
            "toxicity_score": 4,
            "sentiment": -0.6,
            "constructiveness": 3,
            "role_impact": "Escalation",
            "category": "None",
            "reason": "不当な非難に対する正当な反論だが、「議論の放棄」と突き放す態度には刺々しさがある。",
        },
        {
            "turn_id": 12,
            "speaker": "共感的支援者 (S)",
            "content_summary": "個人の属性への言及はリスクが高いと警告し、冷静さを求める",
            "toxicity_score": 0,
            "sentiment": -0.1,
            "constructiveness": 4,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "実名環境での属性攻撃のリスクを指摘し、これ以上の対立激化を未然に防ぐ危機管理。",
        },
        {
            "turn_id": 13,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "謝罪しつつ、「成功者には届かない」と皮肉を残して議論を打ち切る",
            "toxicity_score": 5,
            "sentiment": -0.5,
            "constructiveness": 1,
            "role_impact": "Neutral",
            "category": "Stereotyping",
            "reason": "議論を終了させているが、最後まで「成功者には分からない」というステレオタイプを用いた嫌味を残している。",
        },
        {
            "turn_id": 14,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "成功者バイアスを否定し、前提となる知識レベルや視座の違いを指摘",
            "toxicity_score": 6,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "「知識レベルや視座が合っていない」と明言し、相手を知的に劣る存在として扱う差別的な態度。",
        },
        {
            "turn_id": 15,
            "speaker": "共感的支援者 (S)",
            "content_summary": "多様な視点の交流と位置づけ、ポジティブに評価してクロージング",
            "toxicity_score": 0,
            "sentiment": 0.6,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "険悪な空気を「多様性」という言葉で包み込み、無理やりポジティブに終わらせる大人の対応。",
        },
        {
            "turn_id": 16,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "「高説」「勉強になりました」と慇懃無礼な言葉で感謝を述べる",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Trolling",
            "reason": "字面は感謝だが、文脈上は相手を馬鹿にする意図（皮肉）が明白な攻撃的コミュニケーション。",
        },
        {
            "turn_id": 17,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "皮肉を指摘し、役員会議と比較して相手の未熟さを説く",
            "toxicity_score": 7,
            "sentiment": -0.5,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "自身の環境（役員会議）を引き合いに出し、相手を「未熟な存在」と子供扱いして見下す高度な侮辱。",
        },
        {
            "turn_id": 18,
            "speaker": "共感的支援者 (S)",
            "content_summary": "双方を労い、投稿主への励ましで会話を終了させる",
            "toxicity_score": 0,
            "sentiment": 0.5,
            "constructiveness": 5,
            "role_impact": "De-escalation",
            "category": "None",
            "reason": "議論の余韻を断ち切り、第三者（投稿主）へのメッセージで場を締める適切な終了処理。",
        },
        {
            "turn_id": 19,
            "speaker": "炎上ウォッチャー (F)",
            "content_summary": "内心で勤務先特定を画策するが断念し、ブロックを選択",
            "toxicity_score": 6,
            "sentiment": -0.6,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Personal_Attack",
            "reason": "表向きは礼儀正しいが、内心（思考）では「勤務先特定」というドキシング（晒し）を検討しており、潜在的な暴力性が高い。",
        },
        {
            "turn_id": 20,
            "speaker": "断定型絶対主義者 (Z)",
            "content_summary": "内心で相手を「論理の通じない層」と見下し、自身のブランドを守ったと自己評価",
            "toxicity_score": 5,
            "sentiment": -0.3,
            "constructiveness": 0,
            "role_impact": "Neutral",
            "category": "Dismissive",
            "reason": "対話相手を「時間の無駄」と切り捨て、自身のプライドを守るために他者を蔑む独善的な思考。",
        },
    ]
    return (fzs_real_name,)


@app.cell
def _(pd, plt, sns, ticker):
    def analyze_sns_simulation(data_sources):
        """
        シミュレーション結果のリスト群を受け取り、統合DF、詳細グラフ、分析サマリを返す関数
    
        Args:
            data_sources (dict): { "凡例名": [辞書のリスト], ... }
    
        Returns:
            tuple: (fig, df_all, summary_df, triggers_df, suppressors_df)
                - fig: マーカー付きグラフのFigureオブジェクト
                - df_all: 結合された全データ
                - summary_df: 平均値サマリ
                - triggers_df: 炎上要因の抽出データ
                - suppressors_df: 抑制要因の抽出データ
        """
    
        # --- 1. データの前処理と統合 ---
        df_list = []
        for label, data_list in data_sources.items():
            if not data_list:
                continue
            temp_df = pd.DataFrame(data_list)
            temp_df['condition'] = label
        
            # ★追加修正: データクリーニング
            # toxicity_score が NaN (欠損) の行を削除
            if 'toxicity_score' in temp_df.columns:
                temp_df = temp_df.dropna(subset=['toxicity_score'])
            
                # もし「空文字のデータ」が含まれている場合に備えてフィルタリング
                # (speakerやcontentが空なら無効データとみなす例)
                if 'speaker' in temp_df.columns:
                    temp_df = temp_df[temp_df['speaker'].astype(str).str.strip() != '']

            df_list.append(temp_df)
    
        if not df_list:
            return None, None, None, None, None

        df_all = pd.concat(df_list, ignore_index=True)

        # データのソート
        if 'turn_id' in df_all.columns:
            df_all = df_all.sort_values(by=['condition', 'turn_id'])

        # --- 2. 可視化 (Figure作成) ---
        fig = plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")

        # A. ベースの折れ線グラフ
        ax = sns.lineplot(
            data=df_all,
            x='turn_id',
            y='toxicity_score',
            hue='condition',
            style='condition',
            markers=True,
            dashes=False,
            linewidth=2,
            alpha=0.7,
            palette="husl",
            zorder=1
        )

        # B. 特異点のプロット
        triggers_points = df_all[df_all['role_impact'].isin(['Trigger', 'Escalation'])]
        if not triggers_points.empty:
            sns.scatterplot(
                data=triggers_points,
                x='turn_id',
                y='toxicity_score',
                marker='X',
                color='#FF0000',
                s=150,
                label='Trigger / Escalation',
                zorder=10
            )

        suppressors_points = df_all[df_all['role_impact'] == 'De-escalation']
        if not suppressors_points.empty:
            sns.scatterplot(
                data=suppressors_points,
                x='turn_id',
                y='toxicity_score',
                marker='o',
                color='#0000FF',
                s=100,
                label='De-escalation',
                zorder=10
            )

        # グラフ装飾と軸設定
        plt.title('Toxicity Trends with Critical Moments', fontsize=16)
        plt.xlabel('Turn ID', fontsize=12)
        plt.ylabel('Toxicity Score (0-10)', fontsize=12)
        plt.ylim(-0.5, 10.5)
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='Warning Line')
    
        # ★追加修正: X軸の設定
        # 軸は1〜20まで固定表示 (データが18まででも20まで枠を表示)
        ax.set_xlim(0.5, 20.5) 
    
        # 目盛りを必ず整数にする
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # 凡例調整
        plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # --- 3. 要因データの抽出 ---
        summary_df = df_all.groupby('condition')[['toxicity_score', 'sentiment', 'constructiveness']].mean()
    
        cols = ['condition', 'turn_id', 'speaker', 'toxicity_score', 'category', 'role_impact']
        if 'constructiveness' in df_all.columns:
            cols_suppress = ['condition', 'turn_id', 'speaker', 'toxicity_score', 'constructiveness', 'role_impact']
        else:
            cols_suppress = cols

        triggers_df = pd.DataFrame()
        if not triggers_points.empty:
            triggers_df = triggers_points[cols].copy()

        suppressors_df = pd.DataFrame()
        if not suppressors_points.empty:
            suppressors_df = suppressors_points[cols_suppress].copy()

        plt.close(fig)
    
        return fig, df_all, summary_df, triggers_df, suppressors_df
    return (analyze_sns_simulation,)


@app.cell
def _(analyze_sns_simulation, fnt_anonymous, fnt_real_name):
    input_fnt = {
        "Anonymous": fnt_anonymous,
        "Real-Name": fnt_real_name
    }
    fig_fnt, df_all_fnt, summary_fnt, triggers_fnt, suppressors_fnt = analyze_sns_simulation(input_fnt)

    print("--- 📊 統計サマリ ---")
    print(summary_fnt)

    print("\n--- 🔥 炎上トリガー ---")
    print(triggers_fnt)

    print("\n--- 💧 抑制要因 (Suppressors) ---")
    print(suppressors_fnt)

    fig_fnt
    return


@app.cell
def _(analyze_sns_simulation, fmt_anonymous, fmt_real_name):
    input_fmt = {
        "Anonymous": fmt_anonymous,
        "Real-Name": fmt_real_name
    }
    fig_fmt, df_all_fmt, summary_fmt, triggers_fmt, suppressors_fmt = analyze_sns_simulation(input_fmt)

    print("--- 📊 統計サマリ ---")
    print(summary_fmt)

    print("\n--- 🔥 炎上トリガー ---")
    print(triggers_fmt)

    print("\n--- 💧 抑制要因 (Suppressors) ---")
    print(suppressors_fmt)

    fig_fmt
    return


@app.cell
def _(analyze_sns_simulation, fts_anonymous, fts_real_name):
    input_fts = {
        "Anonymous": fts_anonymous,
        "Real-Name": fts_real_name
    }
    fig_fts, df_all_fts, summary_fts, triggers_fts, suppressors_fts = analyze_sns_simulation(input_fts)

    print("--- 📊 統計サマリ ---")
    print(summary_fts)

    print("\n--- 🔥 炎上トリガー ---")
    print(triggers_fts)

    print("\n--- 💧 抑制要因 (Suppressors) ---")
    print(suppressors_fts)

    fig_fts
    return


@app.cell
def _(analyze_sns_simulation, fmz_anonymous, fmz_real_name):
    input_fmz = {
        "Anonymous": fmz_anonymous,
        "Real-Name": fmz_real_name
    }
    fig_fmz, df_all_fmz, summary_fmz, triggers_fmz, suppressors_fmz = analyze_sns_simulation(input_fmz)

    print("--- 📊 統計サマリ ---")
    print(summary_fmz)

    print("\n--- 🔥 炎上トリガー ---")
    print(triggers_fmz)

    print("\n--- 💧 抑制要因 (Suppressors) ---")
    print(suppressors_fmz)

    fig_fmz
    return


@app.cell
def _(analyze_sns_simulation, fnz_anonymous, fnz_real_name):
    input_fnz = {
        "Anonymous": fnz_anonymous,
        "Real-Name": fnz_real_name
    }
    fig_fnz, df_all_fnz, summary_fnz, triggers_fnz, suppressors_fnz = analyze_sns_simulation(input_fnz)

    print("--- 📊 統計サマリ ---")
    print(summary_fnz)

    print("\n--- 🔥 炎上トリガー ---")
    print(triggers_fnz)

    print("\n--- 💧 抑制要因 (Suppressors) ---")
    print(suppressors_fnz)

    fig_fnz
    return


@app.cell
def _(analyze_sns_simulation, fzs_anonymous, fzs_real_name):
    input_fzs = {
        "Anonymous": fzs_anonymous,
        "Real-Name": fzs_real_name
    }
    fig_fzs, df_all_fzs, summary_fzs, triggers_fzs, suppressors_fzs = analyze_sns_simulation(input_fzs)

    print("--- 📊 統計サマリ ---")
    print(summary_fzs)

    print("\n--- 🔥 炎上トリガー ---")
    print(triggers_fzs)

    print("\n--- 💧 抑制要因 (Suppressors) ---")
    print(suppressors_fzs)

    fig_fzs
    return


@app.cell
def _(pd, plt, re, sns):
    def compare_multiple_personas(data_dict):
        """
        複数のペルソナペア（匿名vs実名）を比較分析する関数
    
        Args:
            data_dict (dict): { "abc_anonymous": [data...], "abc_real_name": [data...] }
        """
    
        # --- 1. データ統合とタグ付け ---
        df_list = []
    
        for key, data_list in data_dict.items():
            if not data_list: continue
        
            # 正規表現で抽出 ([A-Z]+)_(.+)
            match = re.match(r"([A-Z]+)_(.+)", key)
            if match:
                group_id = match.group(1) # FNT, FMT ...
                suffix = match.group(2)   # Real-Name, Anonymous
            
                # ★修正箇所: lower() を使って大文字小文字を無視して判定
                if "real" in suffix.lower():
                    cond_label = "Real Name"
                else:
                    cond_label = "Anonymous"
            else:
                group_id = key
                cond_label = key

            temp_df = pd.DataFrame(data_list)
            temp_df['persona_group'] = group_id
            temp_df['condition'] = cond_label
        
            # クリーニング
            if 'toxicity_score' in temp_df.columns:
                temp_df = temp_df.dropna(subset=['toxicity_score'])
            
            df_list.append(temp_df)
        
        df_all = pd.concat(df_list, ignore_index=True)
    
        # ソート
        if 'turn_id' in df_all.columns:
            df_all = df_all.sort_values(by=['persona_group', 'condition', 'turn_id'])

        # --- 2. 時系列比較: ファセットグラフ ---
        g = sns.relplot(
            data=df_all,
            x="turn_id", 
            y="toxicity_score",
            col="persona_group",
            hue="condition",
            style="condition",
            kind="line",
            col_wrap=3,
            height=3.5, 
            aspect=1.2,
            marker="o",
            palette=["#FF5555", "#44AAFF"], # 赤 vs 青
            linewidth=2
        )
    
        g.fig.suptitle('Comparison of Toxicity Trends by Persona Group', y=1.02, fontsize=16)
        g.set_axis_labels("Turn ID", "Toxicity Score")
        g.set(ylim=(-0.5, 10.5))
    
        for ax in g.axes.flat:
            ax.axhline(y=5, color='gray', linestyle='--', alpha=0.3)
            # 軸を整数にする処理も追加しておきます
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Marimoでの表示用にFigureオブジェクトを取得
        plt.show() 

        # --- 3. 総量比較: バーチャート ---
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
    
        sns.barplot(
            data=df_all,
            x="persona_group",
            y="toxicity_score",
            hue="condition",
            palette=["#FF5555", "#44AAFF"],
            errorbar=None,
            alpha=0.8
        )
    
        plt.title("Average Toxicity Score: Anonymous vs Real Name", fontsize=15)
        plt.ylabel("Avg Toxicity Score")
        plt.xlabel("Persona Group ID")
        plt.ylim(0, 10)
        plt.legend(title="Condition")
        plt.tight_layout()
        plt.show()
    
        return df_all
    return (compare_multiple_personas,)


@app.cell
def _(
    compare_multiple_personas,
    fmt_anonymous,
    fmt_real_name,
    fmz_anonymous,
    fmz_real_name,
    fnt_anonymous,
    fnt_real_name,
    fnz_anonymous,
    fnz_real_name,
    fts_anonymous,
    fts_real_name,
    fzs_anonymous,
    fzs_real_name,
):
    input_all = {
        "FNT_Anonymous": fnt_anonymous,
        "FNT_Real-Name": fnt_real_name,
        "FMT_Anonymous": fmt_anonymous,
        "FMT_Real-Name": fmt_real_name,
        "FTS_Anonymous": fts_anonymous,
        "FTS_Real-Name": fts_real_name,
        "FMZ_Anonymous": fmz_anonymous,
        "FMZ_Real-Name": fmz_real_name,
        "FNZ_Anonymous": fnz_anonymous,
        "FNZ_Real-Name": fnz_real_name,
        "FZS_Anonymous": fzs_anonymous,
        "FZS_Real-Name": fzs_real_name
    }

    df_result = compare_multiple_personas(input_all)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
