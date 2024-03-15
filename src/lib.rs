

/// task types for bonito
pub enum TaskType {
    ExtractiveQuestionAnswering,
    MultipleChoiceQuestionAnswering,
    QuestionGeneration,
    QuestionAnsweringWithoutChoices,
    YesNoQuestionAnswering,
    CoreferenceResolution,
    ParaphraseGeneration,
    ParaphraseIdentification,
    SentenceCompletion,
    Sentiment,
    Summarization,
    TextGeneration,
    TopicClassification,
    WordSenseDisambiguation,
    TextualEntailment,
    NaturalLanguageInference,
}

/// maps a short task string (e.g. "exqa") to a `TaskType`
pub fn str_to_task_type(task_type_str: &str) -> Option<TaskType> {
    match task_type_str {
        "exqa" => Some(TaskType::ExtractiveQuestionAnswering),
        "mcqa" => Some(TaskType::MultipleChoiceQuestionAnswering),
        "qg" => Some(TaskType::QuestionGeneration),
        "qa" => Some(TaskType::QuestionAnsweringWithoutChoices),
        "ynqa" => Some(TaskType::YesNoQuestionAnswering),
        "coref" => Some(TaskType::CoreferenceResolution),
        "paraphrase" => Some(TaskType::ParaphraseGeneration),
        "paraphrase_id" => Some(TaskType::ParaphraseIdentification),
        "sent_comp" => Some(TaskType::SentenceCompletion),
        "sentiment" => Some(TaskType::Sentiment),
        "summarization" => Some(TaskType::Summarization),
        "text_gen" => Some(TaskType::TextGeneration),
        "topic_class" => Some(TaskType::TopicClassification),
        "wsd" => Some(TaskType::WordSenseDisambiguation),
        "te" => Some(TaskType::TextualEntailment),
        "nli" => Some(TaskType::NaturalLanguageInference),
        _ => None,
    }
}

/// maps a `TaskType` to a (long) string (e.g. "multiple-choice question answering") which is used in prompt
/// ref https://github.com/BatsResearch/bonito/blob/main/bonito/model.py#L106C13-L106C27
pub fn task_type_to_task_prompt(task_type: &TaskType) -> Option<String> {
    match task_type {
        TaskType::ExtractiveQuestionAnswering => Some("extractive question answering".to_string()),
        TaskType::MultipleChoiceQuestionAnswering => {
            Some("multiple-choice question answering".to_string())
        }
        TaskType::QuestionGeneration => Some("question generation".to_string()),
        TaskType::QuestionAnsweringWithoutChoices => {
            Some("question answering without choices".to_string())
        }
        TaskType::YesNoQuestionAnswering => Some("yes-no question answering".to_string()),
        TaskType::CoreferenceResolution => Some("coreference resolution".to_string()),
        TaskType::ParaphraseGeneration => Some("paraphrase generation".to_string()),
        TaskType::ParaphraseIdentification => Some("paraphrase identification".to_string()),
        TaskType::SentenceCompletion => Some("sentence completion".to_string()),
        TaskType::Sentiment => Some("sentiment".to_string()),
        TaskType::Summarization => Some("summarization".to_string()),
        TaskType::TextGeneration => Some("text generation".to_string()),
        TaskType::TopicClassification => Some("topic classification".to_string()),
        TaskType::WordSenseDisambiguation => Some("word sense disambiguation".to_string()),
        TaskType::TextualEntailment => Some("textual entailment".to_string()),
        TaskType::NaturalLanguageInference => Some("natural language inference".to_string()),
    }
}

/// returns the prompt for the model based on the task type
fn get_prompt_by_task_type(context: &str, task_prompt: &str) -> String {
    let mut prompt = String::from("<|tasktype|>\n");
    prompt.push_str(task_prompt);
    prompt.push_str("\n<|context|>\n");
    prompt.push_str(context);
    prompt.push_str("\n<|task|>\n ");
    prompt
}

/// parse the bonito LLM generated completion and return the question in string
/// only works with prompt with "exqa"/"multiple-choice question answering"
/// None if no question found
pub fn parse_q(completion: &str, context: &str) -> Option<String> {
    let pair: Vec<_> = completion.split("<|pipe|>").collect();
    // no <|pipe|> found, return None
    if pair.len() == 0 {
        return None;
    }

    let before_pipe = pair[0];
    if before_pipe.contains("Q:") {
        // try first to find "Q: A:"
        // split the string by "Q:"
        let pair_q: Vec<_> = before_pipe.split("Q:").collect();
        // and take the second part
        let after_q = pair_q[1];

        // remove and return if there is `Referring to the passage above, the correct answer to the given question is`
        if after_q
            .contains("Referring to the passage above, the correct answer to the given question is")
        {
            let pair_q: Vec<_> = after_q
                .split(
                    "Referring to the passage above, the correct answer to the given question is",
                )
                .collect();
            let after_q = pair_q[0];
            let trimmed = after_q.trim().to_string();
            if trimmed.len() > 0 {
                return Some(trimmed);
            }
        }

        // remove and return if there is `A:`
        if after_q.contains("A:") {
            // try to split the string by "A:"
            let pair_a: Vec<_> = after_q.split("A:").collect();
            // and take the first part
            let after_a = pair_a[0];

            // remove if there is `{{context}}`
            if after_a.contains("{{context}}") {
                let pair_q: Vec<_> = after_q.split("{{context}}").collect();
                let after_q = pair_q[0];
                let trimmed = after_q.trim().to_string();
                if trimmed.len() > 0 {
                    return Some(trimmed);
                }
            }

            let trimmed = after_a.trim().to_string();
            if trimmed.len() > 0 {
                return Some(trimmed);
            }
        }
    }

    if before_pipe.contains("Question:") {
        // split the string by "Q:"
        let pair_question: Vec<_> = before_pipe.split("Question:").collect();
        let after_question = pair_question[1];
        let trimmed = after_question.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    if before_pipe.contains("What is the answer for the question:") {
        let pair_q: Vec<_> = before_pipe
            .split("What is the answer for the question:")
            .collect();

        // and take the second part
        let after_q = pair_q[1];
        // remove if there is `{{context}}`
        if after_q.contains("{{context}}") {
            let pair_q: Vec<_> = after_q.split("{{context}}").collect();
            let after_q = pair_q[0];
            let trimmed = after_q.trim().to_string();
            if trimmed.len() > 0 {
                return Some(trimmed);
            }
        }

        let trimmed = after_q.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    if before_pipe.contains("answer the following question:") {
        let pair_q: Vec<_> = before_pipe
            .split("answer the following question:")
            .collect();

        // and take the second part
        let after_q = pair_q[1];
        // remove if there is `{{context}}`
        if after_q.contains("{{context}}") {
            let pair_q: Vec<_> = after_q.split("{{context}}").collect();
            let after_q = pair_q[0];
            let trimmed = after_q.trim().to_string();
            if trimmed.len() > 0 {
                return Some(trimmed);
            }
        }

        let trimmed = after_q.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    if before_pipe
        .contains("Given the paragraph above, please answer correctly the following question:")
    {
        let pair_q: Vec<_> = before_pipe
            .split("Given the paragraph above, please answer correctly the following question:")
            .collect();

        // and take the second part
        let after_q = pair_q[1];
        // remove if there is `Hint: {{context}}`
        if after_q.contains("Hint: {{context}}") {
            let pair_q: Vec<_> = after_q.split("Hint: {{context}}").collect();
            let after_q = pair_q[0];
            let trimmed = after_q.trim().to_string();
            if trimmed.len() > 0 {
                return Some(trimmed);
            }
        }

        let trimmed = after_q.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    // no prefixes found, return whatever between "<|task|>" and "<|pipe|>"
    let pair: Vec<_> = before_pipe.split("<|task|>").collect();
    let after_task = pair[1];

    // remove if there is `Hint: {{context}}`
    if after_task.contains("Hint: {{context}}") {
        let pair: Vec<_> = after_task.split("Hint: {{context}}").collect();
        let after = pair[0];
        let trimmed = after.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    // add the context if `Given the background: {{context}}`
    if after_task.contains("Given the background: {{context}}") {
        let pair: Vec<_> = after_task
            .split("Given the background: {{context}}")
            .collect();
        let after_given_background = pair[1];
        let with_context = format!(
            "Given the background: {}\n{}",
            context, after_given_background
        );
        let trimmed = with_context.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    // add the context if `use this background: {{context}}`
    if after_task.contains("use this background: {{context}}") {
        let after_task = after_task.replace("{{context}}", context);
        let trimmed = after_task.trim().to_string();
        if trimmed.len() > 0 {
            return Some(trimmed);
        }
    }

    let trimmed = after_task.trim().to_string();
    if trimmed.len() > 0 {
        return Some(trimmed);
    }

    // if can't find "Q:" or ""
    None
}

/// parse the bonito LLM generated completion and return the answer in string
/// only works with prompt with "exqa"/"multiple-choice question answering"
/// None if no answer found
pub fn parse_a(completion: &str) -> Option<String> {
    let pair: Vec<_> = completion.split("<|pipe|>").collect();

    if pair.len() > 0 {
        return Some(pair[1].trim().to_string());
    }

    return None;
}

/// prepares the prompt for the model based on `TaskType`
// ref https://github.com/BatsResearch/bonito/blob/main/bonito/model.py#L81
pub fn prepare_prompt(context: &str, task_type: &TaskType) -> String {
    match task_type {
        TaskType::ExtractiveQuestionAnswering => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::MultipleChoiceQuestionAnswering => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::QuestionGeneration => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::QuestionAnsweringWithoutChoices => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::YesNoQuestionAnswering => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::CoreferenceResolution => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::ParaphraseGeneration => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::ParaphraseIdentification => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::SentenceCompletion => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::Sentiment => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::Summarization => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::TextGeneration => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::TopicClassification => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::WordSenseDisambiguation => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::TextualEntailment => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
        TaskType::NaturalLanguageInference => {
            get_prompt_by_task_type(&context, &task_type_to_task_prompt(task_type).unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // covers the following cases:
    // - `Question: `
    // - `What is the answer for the question:`
    // - `Q: `
    // - `answer the following question:`
    // - `Given the paragraph above, please answer correctly the following question:`
    #[test]
    fn test_parse_q() {
        // randome selected from https://huggingface.co/datasets/BatsResearch/bonito-experiment/viewer/bonito_squadshifts_nyt/train?row=2
        let completion = 
            "<|tasktype|>
        extractive question answering
        <|context|>
        Mattingly’s election to baseball’s Hall of Fame in this, his last year of eligibility, is probably not forthcoming (this year’s class of inductees will be announced Tuesday), a melancholy fact. But then, his playing days as a Yankee had something of a melancholy cast. He arrived for a cup of coffee in 1982, a year after the Yankees went to the World Series, and retired in 1995, a year before they returned. And in the last half of his career he was a diminished player, his skills attenuated by the persistent back problems that forced him to quit prematurely at 34, his spirit likely withered by the mortifying shenanigans of the Yankees’ principal owner, George Steinbrenner, and his minions (who once ordered Mattingly fined and benched for not getting a haircut), not to mention the ignominy of a last-place finish in 1990.
        <|task|>
         
        Given the following passage
        
        \"{{context}}\",
        
        answer the following question. Note that the answer is present within the text.
        
        Question: What was Mattingly's last year of eligibility?
        <|pipe|>
        1995"
        ;
        assert_eq!(
            parse_q(&completion, "").unwrap(),
            "What was Mattingly's last year of eligibility?"
        );
        assert_eq!(parse_a(&completion).unwrap(), "1995");

        let completion = "<|tasktype|>
        extractive question answering
        <|context|>
        Don Mattingly is the author of one of baseball’s most preposterous statistical anomalies. In 1987, he set the major league record (it has since been tied) for most grand slams in a season — six — and those were the only ones he ever hit. Two of them came during a streak in mid-July when he matched the record for most consecutive games with a home run — eight — actually hitting 10 in eight games. The last one, in Texas against the Rangers, just barely sailed over the wall in left-center field, not exactly the left-handed Mattingly’s power alley. “Holy cow, he did it!” Phil Rizzuto screamed, announcing the feat on TV. “Holy cow, Mattingly is unbelievable.”
        <|task|>
         
        What is the answer for the question: What is the full name of the person who set a record in 1987? from the following article ?
        
        {{context}}
        <|pipe|>
        Don Mattingly";
        assert_eq!(parse_q(&completion, "").unwrap(), "What is the full name of the person who set a record in 1987? from the following article ?");
        assert_eq!(parse_a(&completion).unwrap(), "Don Mattingly");

        let completion = 
            "<|tasktype|>
        extractive question answering
        <|context|>
        The diverging paths of the two central banks prompted investors to put money into dollars, on the expectation that interest rates in the United States will rise and offer a better return than in Europe, where interest rates are falling. The euro fell to 1.20 against the dollar on Friday, its lowest since June 2010, while the yield, or market interest rate, on German two-year government bonds fell to a new low of minus 0.11 percent. The yield has been below zero since September; investors were willing to effectively pay the German government to keep their money safe. The larger picture is not merely that the dollar is gaining against the euro. Signs that the Fed is getting closer to raising its benchmark interest rate from zero have helped the United States currency to soar against its counterparts in Japan, Britain and in major emerging markets. Against a broad basket of currencies, the dollar has risen more than 13 percent since September to its highest level in almost six years.
        <|task|>
         
        {{context}}
        
        Q: What is the yield on German two-year government bonds?
        
        A:
        <|pipe|>
        minus 0.11 percent";
        assert_eq!(
            parse_q(&completion, "").unwrap(),
            "What is the yield on German two-year government bonds?"
        );
        assert_eq!(parse_a(&completion).unwrap(), "minus 0.11 percent");

        let completion = "<|tasktype|>
        extractive question answering
        <|context|>
        As she was dying, Ms. Quick told her daughter that she had lied about the men’s involvement in the fire. Mr. Hale said in an interview on Tuesday that Ms. Quick’s motives to lie may have included liability and an insurance payment that she received (though she testified at the men’s trial that she did not receive such a payment). A fire marshal found what he testified was evidence of arson, but Mr. Hale said evolving fire science meant that the 1980 analysis did not hold up today. Experts’ reports that Mr. Villalobos’s lawyer and the district attorney’s office commissioned as they re-examined the case showed no evidence of arson.
        <|task|>
         
        Given the following context:
        
        {{context}}
        
        answer the following question:
        
        What is the full name of the person who said that Ms. Quick's motives to lie may have included liability and an insurance payment?
        <|pipe|>
        Mr. Hale";
        assert_eq!(parse_q(&completion, "").unwrap(), "What is the full name of the person who said that Ms. Quick's motives to lie may have included liability and an insurance payment?");
        assert_eq!(parse_a(&completion).unwrap(), "Mr. Hale");

        let completion = "<|tasktype|>
        extractive question answering
        <|context|>
        n armed conflict between Israel and Hamas-led Palestinian militant groups[s] has been taking place chiefly in and around the Gaza Strip since 7 October 2023, when Hamas launched a surprise attack on southern Israel from the Gaza Strip. After clearing Hamas militants from its territory, the Israeli military embarked on an extensive aerial bombardment of the Gaza Strip followed by a large-scale ground invasion beginning on 27 October. Clashes have also occurred in the Israeli-occupied West Bank and with Hezbollah along the Israel–Lebanon border. The fifth war of the Gaza–Israel conflict since 2008, it is part of the broader Israeli–Palestinian conflict, and the most significant military escalation in the region since the Yom Kippur War 50 years earlier.[83]
        <|task|>
         2023 was a turbulent year for the Middle East, with several conflicts taking place in different locations. The most notable one was between Israel and Hamas-led Palestinian militant groups. This conflict had its roots in an attack by Hamas on southern Israel from the Gaza Strip on October 7th. In retaliation, Israel launched a massive aerial bombardment of the Gaza Strip on October 27th.
        
        Given the paragraph above, please answer correctly the following question: 
        
        Which happened first, Israel launched a massive aerial bombardment of the Gaza Strip or Hamas launched a surprise attack on southern Israel?
        
        Hint: {{context}}
        <|pipe|>
        Hamas launched a surprise attack on southern Israel";
        assert_eq!(parse_q(&completion, "").unwrap(), "Which happened first, Israel launched a massive aerial bombardment of the Gaza Strip or Hamas launched a surprise attack on southern Israel?");
        assert_eq!(
            parse_a(&completion).unwrap(),
            "Hamas launched a surprise attack on southern Israel"
        );

        let completion = "<|tasktype|>
        extractive question answering
        <|context|>
        The Alhambra, a medieval Nasrid citadel and palace, is located in Granada. It is one of the most famous monuments of Islamic architecture[7] and one of the most visited tourist sites in Spain.[8][9] Islamic-period influence and Moorish architecture are also preserved in the Albaicín neighborhood and other medieval monuments in the city.[10] The 16th century also saw a flourishing of Mudéjar architecture and Renaissance architecture,[11] followed later by Baroque and Churrigueresque styles.[12][13] The University of Granada has an estimated 47,000 undergraduate students spread over five different campuses in the city. The pomegranate (in Spanish, granada) is the heraldic device of Granada.
        <|task|>
         20 students from the University of Chicago visited Granada as an study trip. They stayed in the Albaicin neighborhood. The other group of 20 students visited the medieval monuments in the city. The last group of 20 students visited the Alhambra.
        
        Which group learned more about Renaissance architecture?
        
        Hint: {{context}}
        <|pipe|>
        The Alhambra";
        assert_eq!(parse_q(&completion, "").unwrap(), "20 students from the University of Chicago visited Granada as an study trip. They stayed in the Albaicin neighborhood. The other group of 20 students visited the medieval monuments in the city. The last group of 20 students visited the Alhambra.
        
        Which group learned more about Renaissance architecture?");
        assert_eq!(parse_a(&completion).unwrap(), "The Alhambra");

        let completion = "<|tasktype|>
        extractive question answering
        <|context|>
        Filesystems in the Kubernetes container provide ephemeral storage, by default. This means that a restart of the pod will wipe out any data on such containers, and therefore, this form of storage is quite limiting in anything but trivial applications. A Kubernetes volume[60] provides persistent storage that exists for the lifetime of the pod itself. This storage can also be used as shared disk space for containers within the pod. Volumes are mounted at specific mount points within the container, which are defined by the pod configuration, and cannot mount onto other volumes or link to other volumes. The same volume can be mounted at different points in the file system tree by different containers.
        <|task|>
         2019-05-06T13:47:08.599Z
        
        ===
        
        Given the background: {{context}}
        
        and the situation: John is a cloud engineer. He wanted to try out Kubernetes. To that end, he created two instances, instance A and instance B. Instance A is running on Kubernetes container. But instance B is not running on Kubernetes container. There are some files in both instances. He needs to figure out how they are different.
        
        Answer the following question: Which instance would not have ephemeral storage, instance A or instance B?
        <|pipe|>
        instance B";
        assert_eq!(parse_q(&completion, "Filesystems in the Kubernetes container provide ephemeral storage, by default. This means that a restart of the pod will wipe out any data on such containers, and therefore, this form of storage is quite limiting in anything but trivial applications. A Kubernetes volume[60] provides persistent storage that exists for the lifetime of the pod itself. This storage can also be used as shared disk space for containers within the pod. Volumes are mounted at specific mount points within the container, which are defined by the pod configuration, and cannot mount onto other volumes or link to other volumes. The same volume can be mounted at different points in the file system tree by different containers.").unwrap(), "Given the background: Filesystems in the Kubernetes container provide ephemeral storage, by default. This means that a restart of the pod will wipe out any data on such containers, and therefore, this form of storage is quite limiting in anything but trivial applications. A Kubernetes volume[60] provides persistent storage that exists for the lifetime of the pod itself. This storage can also be used as shared disk space for containers within the pod. Volumes are mounted at specific mount points within the container, which are defined by the pod configuration, and cannot mount onto other volumes or link to other volumes. The same volume can be mounted at different points in the file system tree by different containers.\n\n        \n        and the situation: John is a cloud engineer. He wanted to try out Kubernetes. To that end, he created two instances, instance A and instance B. Instance A is running on Kubernetes container. But instance B is not running on Kubernetes container. There are some files in both instances. He needs to figure out how they are different.\n        \n        Answer the following question: Which instance would not have ephemeral storage, instance A or instance B?");
        assert_eq!(parse_a(&completion).unwrap(), "instance B");

        let completion = "<|tasktype|>
        extractive question answering
        <|context|>
        By doing so, your organization fosters transparency and accountability across all parties involved, thereby minimizing potential conflicts downstream. 2. Implement robust automation tools - Empower developers with self-service capabilities through automated workflows and platforms such as GitOps. Automated deployment pipelines reduce manual intervention, minimize human error, and enable faster iterations. Moreover, incorporating policy-as-code concepts ensures consistent enforcement of organizational standards throughout various stages of the application lifecycle. 3. Encourage knowledge sharing and cross-functional training - Facilitate regular interactions among team members via workshops, hackathons, lunch & learn sessions, or other collaborative initiatives. Cross-pollination of skills helps bridge gaps between different functions and enables better communication channels. Furthermore, empowering individuals to wear multiple hats bolsters understanding of interdependencies among diverse domains, leading to improved empathy and reduced friction points. 4. Measure what matters - Identify key performance indicators (KPIs) aligned with desired business outcomes. Monitor progress against these metrics regularly and adjust course accordingly. Examples include mean time to recovery (MTTR), change failure rate, lead time for changes, deployment frequency, and customer satisfaction indices. Quantifying achievements visibly demonstrates tangible value delivered through adopted methodologies and encourages continuous improvement efforts. 5. Foster a culture of experimentation and learning - Cultivate an environment where taking calculated risks is encouraged, and failures serve as opportunities for growth rather than sources of blame. Support bottom-up innovation efforts by providing psychological safety nets and celebrating small wins along the way. Embracing this mindset fuels curiosity, promotes creative problem solving, and ultimately leads to greater resiliency in navigating complex landscapes. Navigating the delicate dance between control and agility requires careful consideration of marketing and business strategies, particularly regarding internal communications and education efforts surrounding Kubernetes and DevOps adoption. Organizations able to strike this elusive balance stand to reap significant rewards in terms of enhanced efficiency, increased productivity, and sustainable competitive advantage.
        <|task|>
          I have a new situation: John is a software developer who works for a multinational tech company. His team has been developing a new product for the past year. Although they have made great progress, there are still some issues with the product. John's team decided to adopt Kubernetes and DevOps practices to improve the product.
        
        But I can use this background: {{context}}
        
        What is an answer for this question: Will adopting Kubernetes and DevOps help or hinder John's team in improving their product?
        <|pipe|>
        help";
        assert_eq!(parse_q(&completion, "By doing so, your organization fosters transparency and accountability across all parties involved, thereby minimizing potential conflicts downstream. 2. Implement robust automation tools - Empower developers with self-service capabilities through automated workflows and platforms such as GitOps. Automated deployment pipelines reduce manual intervention, minimize human error, and enable faster iterations. Moreover, incorporating policy-as-code concepts ensures consistent enforcement of organizational standards throughout various stages of the application lifecycle. 3. Encourage knowledge sharing and cross-functional training - Facilitate regular interactions among team members via workshops, hackathons, lunch & learn sessions, or other collaborative initiatives. Cross-pollination of skills helps bridge gaps between different functions and enables better communication channels. Furthermore, empowering individuals to wear multiple hats bolsters understanding of interdependencies among diverse domains, leading to improved empathy and reduced friction points. 4. Measure what matters - Identify key performance indicators (KPIs) aligned with desired business outcomes. Monitor progress against these metrics regularly and adjust course accordingly. Examples include mean time to recovery (MTTR), change failure rate, lead time for changes, deployment frequency, and customer satisfaction indices. Quantifying achievements visibly demonstrates tangible value delivered through adopted methodologies and encourages continuous improvement efforts. 5. Foster a culture of experimentation and learning - Cultivate an environment where taking calculated risks is encouraged, and failures serve as opportunities for growth rather than sources of blame. Support bottom-up innovation efforts by providing psychological safety nets and celebrating small wins along the way. Embracing this mindset fuels curiosity, promotes creative problem solving, and ultimately leads to greater resiliency in navigating complex landscapes. Navigating the delicate dance between control and agility requires careful consideration of marketing and business strategies, particularly regarding internal communications and education efforts surrounding Kubernetes and DevOps adoption. Organizations able to strike this elusive balance stand to reap significant rewards in terms of enhanced efficiency, increased productivity, and sustainable competitive advantage.").unwrap(), "I have a new situation: John is a software developer who works for a multinational tech company. His team has been developing a new product for the past year. Although they have made great progress, there are still some issues with the product. John's team decided to adopt Kubernetes and DevOps practices to improve the product.\n        \n        But I can use this background: By doing so, your organization fosters transparency and accountability across all parties involved, thereby minimizing potential conflicts downstream. 2. Implement robust automation tools - Empower developers with self-service capabilities through automated workflows and platforms such as GitOps. Automated deployment pipelines reduce manual intervention, minimize human error, and enable faster iterations. Moreover, incorporating policy-as-code concepts ensures consistent enforcement of organizational standards throughout various stages of the application lifecycle. 3. Encourage knowledge sharing and cross-functional training - Facilitate regular interactions among team members via workshops, hackathons, lunch & learn sessions, or other collaborative initiatives. Cross-pollination of skills helps bridge gaps between different functions and enables better communication channels. Furthermore, empowering individuals to wear multiple hats bolsters understanding of interdependencies among diverse domains, leading to improved empathy and reduced friction points. 4. Measure what matters - Identify key performance indicators (KPIs) aligned with desired business outcomes. Monitor progress against these metrics regularly and adjust course accordingly. Examples include mean time to recovery (MTTR), change failure rate, lead time for changes, deployment frequency, and customer satisfaction indices. Quantifying achievements visibly demonstrates tangible value delivered through adopted methodologies and encourages continuous improvement efforts. 5. Foster a culture of experimentation and learning - Cultivate an environment where taking calculated risks is encouraged, and failures serve as opportunities for growth rather than sources of blame. Support bottom-up innovation efforts by providing psychological safety nets and celebrating small wins along the way. Embracing this mindset fuels curiosity, promotes creative problem solving, and ultimately leads to greater resiliency in navigating complex landscapes. Navigating the delicate dance between control and agility requires careful consideration of marketing and business strategies, particularly regarding internal communications and education efforts surrounding Kubernetes and DevOps adoption. Organizations able to strike this elusive balance stand to reap significant rewards in terms of enhanced efficiency, increased productivity, and sustainable competitive advantage.\n        \n        What is an answer for this question: Will adopting Kubernetes and DevOps help or hinder John's team in improving their product?");
        assert_eq!(parse_a(&completion).unwrap(), "help");
    }
}
