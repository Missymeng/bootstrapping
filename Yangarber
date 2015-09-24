
package dataReuse;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.TreeMap;
import java.math.BigDecimal;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class Yangarber{
	private static String[] commWords;
	
	private static String[] myText;
	
	private static ArrayList<List<CoreLabel>> textList;
	
	private static String[] triggerWords;
	
	private static String[] deletTarget;
	
	
	private static String[] stopWords;
	
	private static String[] contextTriggers;
		
		//all learned words are saved in learnedWordList
		private static ArrayList<String> learnedWordList;
		
		//all learned patterns are reserved in learnedPatterns
		private static ArrayList<String> learnedPatterns;
		
		//each iteration learned entities are saved in learnedTerm[]
		private static String[] learnedTerm;
		
		private static int sentenceNum;
		
		private static HashMap<Integer, Targets> lineTarget;
	
		//all entities extracted based on all patterns
		private static ArrayList<String> negativeEntity;
		private static ArrayList<String> allEntities;
		private static HashMap<String, Entity> entityMap;
		private static HashMap<String, Pattern> patternMap;
		private static ArrayList<String> newPatternMap;
		private static HashMap <String, Pattern> candidatePattern;
		private static String inDir = "/Users/xuelianpan/Documents/workspace/parseXMLFromURL/data/sentences/inputTwoWeeks";
		private static String outDir =  "/Users/xuelianpan/Documents/workspace/parseXMLFromURL/data/sentences/output";
		// maximum number of patterns selected as candidate patterns each time
		private static int numPatterns = 5;
		
		//maximum number of words selected ad learned words each time;
		private static int numEntity = 5;
		
		//maximum number of iterations
		private static int numIterations = 100;
		
		private static double ENTSCORETHRESHOLD = 1.2;
		
		public static void main(String[] args){
			// read in the text that will be used to extract software name
		    String textFile = inDir + "/myText.txt";
		    myText = readInContent(myText, textFile);
			System.out.println("@@@@@@@@" + myText.length);
			
			 // remove punctuation from myText.txt
			//removeStopWords();
			//System.out.println("The punctuations have been removed from myText.txt!");
			
			// the number of sentence
			sentenceNum = myText.length;
			
			//tokenize the text
			textList = new ArrayList<List<CoreLabel>>();
			for(int i = 0; i < sentenceNum; i++){
				textList.add(getTokens(myText[i]));
			}
			
			// read in common words String[] commWords
			String commonFile = inDir + "/commonwords.txt";
			commWords = readInContent(commWords, commonFile);
			
			
			//read in trigger words String[] triggerWords;
			String trigFile = inDir + "/triggerwords.txt";
			triggerWords = readInContent(triggerWords, trigFile);
			
			
			//read in delete targets String[] deletTarget
			String deletFile = inDir + "/target.txt";
			deletTarget = readInContent(deletTarget, deletFile);
			
		
			
			//read in stop words list String[] stopWords;
			String stopFile =  inDir + "/stopwords.txt";
			stopWords = readInContent(stopWords, stopFile);
			
			//read in context trigger word list String[] contextTriggers
			String conTrigFile = inDir + "/contextTriggers.txt";
			contextTriggers = readInContent(contextTriggers, conTrigFile);
			
			
			
			processText();
			
			System.out.println("Finish!");	
			//printTest("Finish!");
		
		}
		
		private static String[] readInContent(String[] text, String fileName){
			Database db = new Database(fileName);
			Iterator<String> it = db.getIterator();
			text = new String[db.getSize()];
			for(int i = 0; it.hasNext(); i++){
				String str = it.next().trim();
				text[i] = str;	
			}
			
			return text;
		}
		
		private static void printOutCandidatePattern(int iterationNum){
			String entityFile = outDir + "/candidatePattern.txt";
			try{
				PrintWriter wr = new PrintWriter(new FileWriter(entityFile, true));
				//print out learned entities
				wr.println("****************This is " + iterationNum + "times extractions*************");
				
				// print out candidate pattern
				//wr.println("This is candidate patterns");
				Iterator<String> candPatIt = candidatePattern.keySet().iterator();
				while(candPatIt.hasNext()){
					String key = candPatIt.next();
					Pattern candPat = candidatePattern.get(key);
					
					Iterator<String> enIt = candPat.getExtWords();
					String result = "";
					while(enIt.hasNext()){
						result += enIt.next() + ";";
					}
					
					Iterator<String> negIt = candPat.getNegWords();
					String negResult = "";
					while(negIt.hasNext()){
						negResult += negIt.next() + ";";
					}
					
					Iterator<String> posIt = candPat.getPosWods();
					String posReuslt = "";
					while(posIt.hasNext()){
						posReuslt += posIt.next() + ";";
					}
					wr.println(key + "##: " + candPat.getPatScore() + 
							 ", extracted positive entities: " + posReuslt);
					
					 wr.println(key + "##: " + candPat.getPatScore() + 
							 ", extracted unlabeled entities: " + result);
					 
					 wr.println(key + "##: " + candPat.getPatScore() + 
							 ", extracted negative entities: " + negResult);
					 
				}
				wr.close();
			 }catch(Exception ex){
				   ex.printStackTrace();
			}
			
		}
		private static void printOutLearnedWord(int iterationNum){
			String entityFile = outDir + "/mylearnedEnity.txt";
			try{
				PrintWriter wr = new PrintWriter(new FileWriter(entityFile, true));
				//print out learned entities
				wr.println("****************This is " + iterationNum + "times extractions*************");
				
				//wr.println("This is learned entities");
				String str = "";
				for (int i = 0; i < learnedWordList.size(); i++){
					str += learnedWordList.get(i) + ",";
				}
				
				int index = str.lastIndexOf(",");
				str = str.substring(0, index);
				wr.println(str);
				wr.println();
				
				wr.close();
			 }catch(Exception ex){
				   ex.printStackTrace();
			}
			
		}
		
		private static void printOutTestMap(int iterationNum){
			
			String entityFile = outDir + "/outMapTest.txt";
			try{
				
				PrintWriter wr = new PrintWriter(new FileWriter(entityFile, true));
				//print out learned entities
				wr.println("************This is " + iterationNum + "times extractions***********");
				
				//print out entityMap
				wr.println("This is etityMap!");
				//Iterator<String> myIt = entityMap.keySet().iterator();
				
				for(int i = 0; i < learnedTerm.length; i++){
					String key = learnedTerm[i];
					if (key != null){
						Entity ent = entityMap.get(key);
						Iterator<String> entIt = ent.getPatterns();
						String result = "";
						while(entIt.hasNext()){
							String tempPatKey = entIt.next();
							Pattern tempPat = patternMap.get(tempPatKey);
							result += tempPatKey + ", pattern score is " + tempPat.getPatScore() + "; ";
						}
						int index = result.lastIndexOf(";");
						result = result.substring(0, index);
						 wr.println(key + "%%: " + result + ", entity score is # " + ent.getScore());
						 wr.println(key + ": hasUpperCase = " + ent.getUpCase() + 
								 "; hasVersionNumber = " + ent.getVersionNumber() + 
								 "; LeftContextTrigger = " + ent.getLeftTrigger() +
								 "; RightContextTrigger = " + ent.getRightTrigger());
				    }
					
				}
				
					
				wr.println("***************This is split line******************");
				
				//print out patternMap
				wr.println("This is patternMap!");
				//Iterator<String> pattIt = patternMap.keySet().iterator();
				if(candidatePattern.size() > 0){
					Iterator<String> pattIt = candidatePattern.keySet().iterator();
					while(pattIt.hasNext()){
						String key = pattIt.next();
						//postive entity
						//Iterator<String> posIt = patternMap.get(key).getPosWods();
						Iterator<String> posIt = candidatePattern.get(key).getPosWods();
						String posStr = "";
						while(posIt.hasNext()){
							posStr += posIt.next() + ";";
						}
						
						//extracted entity
						//Iterator<String> exIt = patternMap.get(key).getExtWords();
						Iterator<String> exIt = candidatePattern.get(key).getExtWords();
						String exStr = "";
						while(exIt.hasNext()){
							exStr += exIt.next() + ";";
						}
						
						//negative entity
						//Iterator<String> negIt =  patternMap.get(key).getNegWords();
						Iterator<String> negIt = candidatePattern.get(key).getNegWords();
						String negStr = "";
						while(negIt.hasNext()){
							negStr += negIt.next() + ";";
						}
						
						//wr.println(key + "$$: "+ patternMap.get(key).getPatScore() + ", postive entity: " + posStr);
						wr.println(key + "$$: "+ candidatePattern.get(key).getPatScore() + ", postive entity: " + posStr);
						wr.println(key + "$$ unlabled entity: " + exStr);
						wr.println(key + "$$ negative entity: " + negStr);
						wr.println("#######");
				}
				
			}  
				wr.close();
			 }catch(Exception ex){
				   ex.printStackTrace();
			}
		}
		  private static void printOutTest(String line, String patStr, String targStr){
				String entityFile = outDir + "/outTest.txt";
				try{
					PrintWriter wr = new PrintWriter(new FileWriter(entityFile, true));
						
					    wr.println("Text: " + line);
					    wr.println("&&&&&&&&" + patStr +":" + targStr);
						wr.println("#######");
				
					wr.close();
				 }catch(Exception ex){
					   ex.printStackTrace();
				}
			}

		private static void printTestTime(int iterationNum, String section, double time){
			String timeFile =  outDir +"/time.txt";
			try{
				PrintWriter wr = new PrintWriter(new FileWriter(timeFile, true));
					
				    wr.println("The #" + iterationNum + " iteration, " + section + " take  ##" + time + " seconds");
				   
					wr.println("**********************");
			
				wr.close();
			 }catch(Exception ex){
				   ex.printStackTrace();
			}
		}
		private static void processText(){
			//learned words list.
			learnedWordList = new ArrayList<String>();
			negativeEntity = new ArrayList<String>();
			
			learnedPatterns = new ArrayList<String>();
			patternMap = new HashMap<String, Pattern>();
			newPatternMap = new ArrayList<String>();
			
			//Initialize HashMap<String, Entity> entityMap
			entityMap = new HashMap<String, Entity>();
					
			// read in the seed into the system
			String seedFile = inDir + "/seed.txt";
			Database db = new Database(seedFile);
			Iterator<String> it = db.getIterator();
			String[] seeds = new String[db.getSize()];
			for (int i = 0; it.hasNext(); i++){
				String str = it.next().trim();
				seeds[i] = str;
				learnedWordList.add(str);
			}
			
			//Stopwatch timer = new Stopwatch();
			// label the text based on the words in the seed file
			labelSoftware(seeds);
			System.out.println("The seed words have been labeled in myText.txt!");
			//double labelTime = timer.elapsedTime();
			//printTestTime(0, "Labeling time", labelTime);
			
			//timer = new Stopwatch();
			// get patterns based on seed words
			allEntities = new ArrayList<String>();
			for (int i = 0; i < learnedWordList.size(); i++){
				allEntities.add(learnedWordList.get(i));
			}
			Iterator<Integer> lineIt = lineTarget.keySet().iterator();
			while(lineIt.hasNext()){
				int n = lineIt.next();
				getCandidatePattern(n, seeds);
				
			}
			//labelTime = timer.elapsedTime();
			//printTestTime(0, "Generating patterns", labelTime);
			
			//timer = new Stopwatch();
			//<software> X </software> is used to replace the special string such as <software> SPSS </software>
			lineIt = lineTarget.keySet().iterator();
			while(lineIt.hasNext()){
				int n = lineIt.next();
				replaceTargetString(n, seeds);
			}
			//labelTime = timer.elapsedTime();
			//printTestTime(0, "Replacing targets", labelTime);
			
			//timer = new Stopwatch();
			// calculate pattern score
			calculatePattScore(numPatterns);
			
			//labelTime = timer.elapsedTime();
			//printTestTime(0, "Calculating pattern score", labelTime);
			
			// use the patterns extracted by seeds to extract new targets
			
			//timer = new Stopwatch();
			Iterator<String> patternIt = candidatePattern.keySet().iterator();
			while(patternIt.hasNext()){
				String key = patternIt.next();
				extractWords(key);	
			}
			//labelTime = timer.elapsedTime();
			//printTestTime(0, "Extracting words", labelTime);
			
			// get entities based on patterns
			//timer = new Stopwatch();
			
			calculateWordScore();
			
			//labelTime = timer.elapsedTime();
			//printTestTime(0, "Calculating entity score", labelTime);
			
			//timer = new Stopwatch();
			printOutTestMap(0);
			printOutCandidatePattern(0);
			printOutLearnedWord(0);
			//labelTime = timer.elapsedTime();
			//printTestTime(0, "Printing results", labelTime);

			//Bootstrapping
			for (int i = 1; i < numIterations; i++){
				System.out.println("The " + i +"th iteration!");
				
			     
				// label the text based on the words in the seed file
				if(learnedTerm[0] != null){
					//timer = new Stopwatch();
					labelSoftware(learnedTerm);
					//labelTime = timer.elapsedTime();
					//printTestTime(i, "Labeling entity score", labelTime);
					
					//timer = new Stopwatch();
					removeSeedFromUnlabledEntMap(learnedTerm);
					//labelTime = timer.elapsedTime();
					//printTestTime(i, "RemoveSeedFromUnlabledEntMap", labelTime);
					
					//timer = new Stopwatch();
					lineIt = lineTarget.keySet().iterator();
					while(lineIt.hasNext()){
						int n = lineIt.next();
						getCandidatePattern(n, learnedTerm);
					}
					//labelTime = timer.elapsedTime();
					//printTestTime(i, "Generating patterns", labelTime);
					
					//timer = new Stopwatch();
					lineIt = lineTarget.keySet().iterator();
					while(lineIt.hasNext()){
						int n = lineIt.next();
						replaceTargetString(n, learnedTerm);
					}
					//labelTime = timer.elapsedTime();
					//printTestTime(i, "Replacing targets", labelTime);
					
				 }
				
				//timer = new Stopwatch();	
					// calculate pattern score
				calculatePattScore(numPatterns);
				//labelTime = timer.elapsedTime();
				//printTestTime(i, "Calculating pattern score", labelTime);
				
				//timer = new Stopwatch();
				patternIt = candidatePattern.keySet().iterator();
				while(patternIt.hasNext()){
					String key = patternIt.next();
					if(newPatternMap.contains(key)){
						extractWords(key);
					}
				}
				//labelTime = timer.elapsedTime();
				//printTestTime(i, "Extracting words", labelTime);
				
					// get entities based on patterns
				//timer = new Stopwatch();
				calculateWordScore();
				//labelTime = timer.elapsedTime();
				//printTestTime(i, "Calculating entity score", labelTime);
				
				//timer = new Stopwatch();
				printOutTestMap(i);
				printOutCandidatePattern(i);
				printOutLearnedWord(i);
				
				//labelTime = timer.elapsedTime();
				//printTestTime(i, "Printing results", labelTime);
				
				
				if (candidatePattern.size() == 0){
					break;
				}
				
			}
			
			// print out the marked text
			printOutMarkedText();
			
		}
		
	private static void replaceTargetString(int n, String[] targterms){
		String myStr = myText[n];
		for(int i = 0; i < targterms.length; i++){
			String targTerm = targterms[i];
			String targStr = "<software> " + targTerm + " </software>";
			int index = myStr.indexOf(targStr);
			while(index >= 0){
				myStr = myStr.replace(targStr, "<software> X </software>");
				index = myStr.indexOf(targStr);
			}
			
			
		}
		myText[n] = myStr;
		textList.set(n, getTokens(myStr));
		
		
	}
		
		private static void removeSeedFromUnlabledEntMap(String[] seeds){
			for(int i = 0; i < learnedTerm.length; i++){
				String seed = seeds[i];
				Iterator<String> patIt = patternMap.keySet().iterator();
				while(patIt.hasNext()){
					String key = patIt.next();
					Pattern pattern = patternMap.get(key);
					pattern.removeUnlabeledEntity(seed);
					patternMap.put(key, pattern);
					
				}
			}
		}
		
		private static boolean isSingleLowcase(String target, double leftConHasTrig, double rightConHasTrig){
			boolean flag = false;
			if(target.equals(target.toLowerCase()) && leftConHasTrig != 1 && rightConHasTrig != 1){
				flag = true;
			}
			return flag;
		}
		
		
		
		/**
		 * Calculate the score of extracted words by candidate patterns
		 */
		private static void calculateWordScore(){
			
			//candidate word list
			HashSet<String> wordSet = new HashSet<String>();
			Iterator<String> candIt = candidatePattern.keySet().iterator();
			while (candIt.hasNext()){
				String patKey = candIt.next();
				Pattern patValue = candidatePattern.get(patKey);
				Iterator<String> candPatIt = patValue.getExtWords();
				while(candPatIt.hasNext()){
					wordSet.add(candPatIt.next());
				}
				
			}
		
			//calculate the score of words
			HashMap<String, Double> candidateWords = new HashMap<String, Double>();
			ValueComparator comWords =  new ValueComparator(candidateWords);
			TreeMap<String,Double> sortedWords = new TreeMap<String,Double>(comWords);
			
			Iterator<String> wordIt = wordSet.iterator();
			while(wordIt.hasNext()){
				String term = wordIt.next();
				Entity ent = entityMap.get(term);
				int size = ent.countPatterns();
				if (size > 1){
					Iterator<String> patList = ent.getPatterns();
					double total = 1.0;
					while(patList.hasNext()){
						//patList.next is the pattern that extracted this entity;
						String temp = patList.next();
						Pattern myPat = patternMap.get(temp);
						//total += myPat.getPatScore();
						total = (double)total * (1 - myPat.getConfScore());
								
					}
					double score = 1 - total;
					
					
					ent.setScore(score);
					candidateWords.put(term, score);
					entityMap.put(term, ent);
					
				}
				
			}
			sortedWords.putAll(candidateWords);
			
			// identify learned words base on word score
			learnedTerm = new String[numEntity];
			Iterator<String> myIt = sortedWords.keySet().iterator();
			int i = 0;
			while(myIt.hasNext() && i < numEntity){
				String myKey = myIt.next();
				double score = candidateWords.get(myKey);
				/*if (!learnedWordList.contains(myKey) && score >= ENTSCORETHRESHOLD){
					learnedWordList.add(myKey);
					learnedTerm[i] = myKey;
					i++;
				}*/
					
				if(!learnedWordList.contains(myKey)){
					learnedWordList.add(myKey);
					learnedTerm[i] = myKey;
					i++;
				}
			
				
			}
					
		}
		
		
		
		
		private static void printOutMarkedText(){
			String markedFile = outDir + "/markedText.txt";
			try{
				PrintWriter wr = new PrintWriter(new FileWriter(markedFile));
				for(int i = 0; i < myText.length; i++){
					wr.println("#" + i + " " + myText[i]);
				}
				
			
				wr.close();
			}catch(Exception ex){
				   ex.printStackTrace();
			  }
		}
	
		
	
	 	
		
	 	/**
	 	 * Calculate pattern score and select top numPatterns patterns to HashMap<String, Pattern> candidatePattern
	 	 */
		private static void calculatePattScore(int numberOfPatt){
			newPatternMap = new ArrayList<String>();
			candidatePattern = new HashMap<String, Pattern>();
			HashMap<String, Double> myScore = new HashMap<String, Double>();
			ValueComparator comMe =  new ValueComparator(myScore);
			TreeMap<String,Double> sortedMe = new TreeMap<String,Double>(comMe);
			
			Iterator<String> patIt = patternMap.keySet().iterator();
			while (patIt.hasNext()){
				String pattStr = patIt.next();
				Pattern pat = patternMap.get(pattStr);
				int posNum = pat.countPosWords();
				int negNum = pat.countNegWords();
				int unlabedNum = pat.countExtWords();
				double t = 0.0;
				
				double acc = 0;
				double conf = 0;
				
				if(unlabedNum == 0){
					if(!learnedPatterns.contains(pattStr)){
						acc = (double)posNum/(double)(posNum + negNum);
						conf = acc;
					}else{
						acc = (double)posNum/(double)(posNum + negNum);
						conf = (double)posNum/(double)(posNum + negNum + unlabedNum);
					}
					 
				}else{
					
					acc = (double)posNum/(double)(posNum + negNum);
					conf = (double)posNum/(double)(posNum + negNum + unlabedNum);	
						
				}
				
				if(acc >= 0.8){
					t = (double)conf * Math.log(posNum);
				}
				pat.setConfScore(conf);	
				pat.setPatScore(t);
				patternMap.put(pattStr, pat);
				myScore.put(pattStr, t);
				
			}
			
			sortedMe.putAll(myScore);
			
			Iterator<String> myIterator = sortedMe.keySet().iterator();
			int i = 0;
			while(myIterator.hasNext() && i < numberOfPatt){
				String key = myIterator.next();
				Pattern pat = patternMap.get(key);
				int unlabeledNum = pat.countExtWords();
				double score = pat.getPatScore();
				if(score > 0){
					if((unlabeledNum == 0 && !learnedPatterns.contains(key))||unlabeledNum != 0){
						candidatePattern.put(key, pat);
						if(!learnedPatterns.contains(key)){
							learnedPatterns.add(key);
							newPatternMap.add(key);
						}
						i++;
					}
					
				}
			
			}
			
		}
		
		
		
		
	
		
		private static void extractWords(String patStr){
			
			if (patStr.startsWith("<>")){
				extraPreWords(patStr);
			}else if(patStr.endsWith("<>")){
				extraNexWords(patStr);
			}else{
				extraMidWords(patStr);
			}
			
			/*if (patStr.startsWith("<>")){
				extraPreWords(patStr);
			}else{
				extraNexWords(patStr);
			}*/
			
			
			
			
	    }
		
		
		private static String clearMidEntity(String targStr, String pattern, 
				List<CoreLabel>tokens, int firstIndex, int lastIndex){
			String result = "";
				List<CoreLabel> targetTerm = getTokens(targStr);
				if(targetTerm.size()<= 5){
					//start is the end index of pattern
					String conFirst = tokens.get(firstIndex).get(LemmaAnnotation.class);
					String conLast = tokens.get(lastIndex).get(LemmaAnnotation.class);
					String lastTerm = targetTerm.get(targetTerm.size()-1).get(LemmaAnnotation.class);
					
					//extracted entity has upperCase letter? 1 means have upperCase letter; 0.5 means no.
					double hasUpCase = 0.6;
					if(!targStr.equals(targStr.toLowerCase())){
						hasUpCase = 1.0;
					}
					
					//extracted entity has version number? 1 means have version number; 0.5 means no
					double hasNumber = 0.2;
					
					// result is the formated extracted entity
					result = formatNewEntity(targStr, targetTerm);
					for (int i = targetTerm.size() - 1; i >= 0; i--){
						String str = targetTerm.get(i).get(TextAnnotation.class);
						String pos = targetTerm.get(i).get(PartOfSpeechAnnotation.class);
						if(isVersionStr(str, pos)){
							hasNumber = 1.0;
							break;
						}
					}
					
					//context of extracted entity has extracting trigger words, such as software, package, and tool which is in triggerwords.txt
					// or has deleting trigger words, such as function, assay, and database which is target.txt?
					// 1 means has extracting trigger words, 0 means has deleting trigger words, 0.5 means has none of both
					double leftConHasTrig = 0.2;
					double rightConHasTrig = 0.2;
					if(isTarget(triggerWords, lastTerm)||isTarget(triggerWords, conFirst)||isTarget(triggerWords, conLast)){
						leftConHasTrig = 1.0;
					}
					if(isTarget(deletTarget, conFirst) || isTarget(deletTarget, lastTerm)||isTarget(deletTarget, conLast)){
						leftConHasTrig = -0.4;
					}
					
					if(lastIndex == tokens.size() - 1){
						if(isTarget(triggerWords, conLast)){
							rightConHasTrig = 1.0;
						}
						if(isTarget(deletTarget, conLast)){
							rightConHasTrig = -0.4;
						}
					}
					
					if(lastIndex < tokens.size() - 1){
						String secStr = tokens.get(lastIndex + 1).get(TextAnnotation.class);
						if(isTarget(triggerWords, conLast)){
							rightConHasTrig = 1.0;
						}
						if(isTarget(deletTarget, conLast)||isTarget(deletTarget, secStr)){
							rightConHasTrig = -0.4;
						}
						
					}
					
					
				
						int index = result.indexOf(" ");
						if(index < 0){
							if(!result.equals(result.toLowerCase())){
								putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
							}else{
								String myPos = tokens.get(firstIndex + 1).get(PartOfSpeechAnnotation.class);
								
								if(isTarget(commWords, result) || !myPos.startsWith("N") 
										|| isSingleLowcase(result, leftConHasTrig, rightConHasTrig)){
									result = clearNegEntity(result);
									Pattern pat = patternMap.get(pattern);
									pat.addNegWords(result);
									patternMap.put(pattern, pat);
									result = "";
								
									
								}else{
									putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
									
								}
							
							}
						}else{
							putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
						}
					//}
				}	
				
			return result;
		
			
		}
		
		private static void extraMidWords(String patStr){
			String[] midCons = patStr.split(" ");
			String first = midCons[0];
			String last = midCons[2];
			
			for (int i = 0; i < sentenceNum; i++){
				//String line = myText[i];
				List<CoreLabel> tokens = textList.get(i);
				int tokenSize = tokens.size();
				
				for (int j = 0 ; j < tokenSize - 1; j++){
					boolean flag = false;
					int lastInd = -1;
					String firstStr = tokens.get(j).get(TextAnnotation.class);
					if ((j == 0 ||(j > 0 && firstStr.equals(firstStr.toLowerCase()))) && 
							tokens.get(j).get(LemmaAnnotation.class).equalsIgnoreCase(first) &&
							tokenSize - j >= 3){
						int nextBreak = getNextPoint(tokens, j, tokenSize);
						if (nextBreak - j - 1 >= 2){
							String mid = tokens.get(j + 1).get(TextAnnotation.class);
							String midPos = tokens.get(j + 1).get(PartOfSpeechAnnotation.class);
							lastInd = findLastWord(tokens, j+1, nextBreak, last);
							if(lastInd != -1){
								if (!mid.equals(mid.toLowerCase())){
									flag = true;
									for (int k = j + 2; k < lastInd; k++){
										String temp = tokens.get(k).get(TextAnnotation.class);
										String pos = tokens.get(k).get(PartOfSpeechAnnotation.class);
										if ((temp.equals(temp.toLowerCase()) && !isValidWord(temp, pos))){
											flag = false;
											break;
										}
									}
								}/*else{
									if(lastInd - j > 2 || !midPos.startsWith("N")){
										flag = false;
									}else{
										flag = true;
									}
									
								}*/
							}
						}
					}
					
					if (flag == true){
						String targStr = "";
						for (int n = j+1; n < lastInd; n++){
							targStr += tokens.get(n).get(TextAnnotation.class) + " ";		
						}
						
						targStr = targStr.trim();
						if (!targStr.equals("") && targStr.length() >= 3){
							String[] preTerm = targStr.trim().split(" ");
							if(preTerm.length > 1){
								targStr = clearMidEntity(targStr, patStr, tokens, j, lastInd);
								//extracted entity is added to patternMap's unlabeled entity arrayList
								if(!targStr.equals("")){
									Pattern pat = patternMap.get(patStr);
									if(!learnedWordList.contains(targStr)){
										pat.addExtWords(targStr);
									}else{
										pat.addPosWords(targStr);
									}
									patternMap.put(patStr, pat);
									
								}
							}else{
								   if(!targStr.equals(targStr.toLowerCase())){
									   targStr = clearMidEntity(targStr, patStr, tokens, j, lastInd);
										if(!targStr.equals("")){
											Pattern pat = patternMap.get(patStr);
											if(!learnedWordList.contains(targStr)){
												pat.addExtWords(targStr);
											}else{
												pat.addPosWords(targStr);
											}
											patternMap.put(patStr, pat);	
										}
										
								   }else{
									   String targetPos =  tokens.get(j+1).get(PartOfSpeechAnnotation.class);
									   if(!targetPos.startsWith("N")||isTarget(commWords, targStr)){
											targStr = clearNegEntity(targStr);
											//extracted entity is added to patternMap's negative entity arrayList
											Pattern pat = patternMap.get(patStr);
											pat.addNegWords(targStr);
											patternMap.put(patStr, pat);
										}else{
											targStr = clearMidEntity(targStr, patStr, tokens, j, lastInd);
											if(!targStr.equals("")){
												Pattern pat = patternMap.get(patStr);
												if(!learnedWordList.contains(targStr)){
													pat.addExtWords(targStr);
												}else{
													pat.addPosWords(targStr);
												}
												patternMap.put(patStr, pat);
												
											}
										}
								   }
			
								}
							}
							
							//printOutTest(line, patStr, targStr);	
							j = lastInd;
							
						}
					
						
						
					}
				}
			}

		
		private static int findLastWord(List<CoreLabel> tokens, int start, int end, String word){
			int index = -1;
			for (int i = start + 1; i < end; i++){
				String temp = tokens.get(i).get(TextAnnotation.class);
				if (temp.equals(temp.toLowerCase()) && tokens.get(i).get(LemmaAnnotation.class).equalsIgnoreCase(word)){
					index = i;
					break;
				}
			}
			
			return index;
		}
		
		private static boolean inTargetEntity(List<CoreLabel> tokens, int start, int end){
			boolean flag = false;
			
			if (getIndex(tokens, "<software>") < 0){
				flag = false;
			}else{
				ArrayList<Integer> sIndex = new ArrayList<Integer>();
				ArrayList<Integer> eIndex = new ArrayList<Integer>();
				for (int i = 0; i < tokens.size(); i++){
					String myToken = tokens.get(i).get(TextAnnotation.class);
					if (myToken.equals("<software>")){
						sIndex.add(i);
					}
					if(myToken.equals("</software>")){
						eIndex.add(i);
					}
				}
				
				for (int j = 0; j < sIndex.size(); j++){
					int s = sIndex.get(j);
					int e = eIndex.get(j);
					if (s < start && e > end){
						flag = true;
						break;
					}
				}
				
			}
				
			return flag;
		}
		
		private static String clearNextPosEntity(String targStr, String pattern, 
				List<CoreLabel>tokens, int start,int patLength){
			   String result = "";
				List<CoreLabel> targetTerm = getTokens(targStr);
				if(targetTerm.size() <= 5){
				//start is the end index of pattern
				String conFirst = tokens.get(start).get(LemmaAnnotation.class);
				String conSecond = tokens.get(start - 1).get(LemmaAnnotation.class);
				String lastTerm = targetTerm.get(targetTerm.size() - 1).get(LemmaAnnotation.class);
			/*	if(isTarget(deletTarget, lastTerm)){
					result = clearNegEntity(targStr);
					Pattern pat = patternMap.get(pattern);
					pat.addNegWords(result);
					patternMap.put(pattern, pat);
					result = "";
				}else{*/
					//extracted entity has upperCase letter? 1 means have upperCase letter; 0.5 means no.
					double hasUpCase = 0.6;
					if(!targStr.equals(targStr.toLowerCase())){
						hasUpCase = 1.0;
					}
					
					//extracted entity has version number? 1 means have version number; 0.5 means no
					double hasNumber = 0.2;
					
					// result is the formated extracted entity
					result = formatNewEntity(targStr, targetTerm);
					for (int i = targetTerm.size() - 1; i >= 0; i--){
						String str = targetTerm.get(i).get(TextAnnotation.class);
						String pos = targetTerm.get(i).get(PartOfSpeechAnnotation.class);
						if(isVersionStr(str, pos)){
							hasNumber = 1.0;
							break;
						}
					}
					
					//context of extracted entity has extracting trigger words, such as software, package, and tool which is in triggerwords.txt
					// or has deleting trigger words, such as function, assay, and database which is target.txt?
					// 1 means has extracting trigger words, 0 means has deleting trigger words, 0.5 means has none of both
					
					double leftConHasTrig = 0.2;
					double rightConHasTrig = 0.2;
					if(isTarget(triggerWords, lastTerm) || isTarget(triggerWords, conFirst)){
						leftConHasTrig = 1.0;
					}
					
					if(isTarget(deletTarget, conFirst)||isTarget(deletTarget, conSecond)||isTarget(deletTarget, lastTerm)){
						leftConHasTrig = -0.4;
					}
					
					if(start + targetTerm.size() + 1 <= tokens.size() - 1){
						String preFirst = tokens.get(start + targetTerm.size() + 1).get(LemmaAnnotation.class);
						if(start + targetTerm.size() + 1 == tokens.size() - 1){
							if(isTarget(triggerWords, preFirst)){
								rightConHasTrig = 1.0;
							}
							if(isTarget(deletTarget, preFirst)){
								rightConHasTrig = -0.4;
							}
						}else if(start + targetTerm.size() + 1 == tokens.size() - 2){
							String preSecond = tokens.get(start + targetTerm.size() + 2).get(LemmaAnnotation.class);
							if(isTarget(triggerWords, preFirst)){
								rightConHasTrig = 1.0;
							}
							if(isTarget(deletTarget, preFirst) || isTarget(deletTarget, preSecond)){
								rightConHasTrig = -0.4;
							}
						}else{
							String preSecond = tokens.get(start + targetTerm.size() + 2).get(LemmaAnnotation.class);
							String preThird = tokens.get(start + targetTerm.size() + 3).get(LemmaAnnotation.class);
							if(isTarget(triggerWords, preFirst)){
								rightConHasTrig = 1.0;
							}
							if(isTarget(deletTarget, preFirst) || isTarget(deletTarget, preSecond) || isTarget(deletTarget, preThird)){
								rightConHasTrig = -0.4;
							}
						}
					}
					
					
						int index = result.indexOf(" ");
						if(index < 0){
							String myPos = tokens.get(start + 1).get(PartOfSpeechAnnotation.class);
							if(isTarget(commWords, result) || !myPos.startsWith("N") 
									|| isSingleLowcase(result, leftConHasTrig, rightConHasTrig)){
								result = clearNegEntity(result);
								Pattern pat = patternMap.get(pattern);
								pat.addNegWords(result);
								patternMap.put(pattern, pat);
								result = "";
								
							}else{
								putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
								
							}
								
						}else{
							putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
						}
				
			}
		
			return result;
			
		}
		
		private static void extraNexWords(String patStr){
			String nexCon = patStr.substring(0, patStr.indexOf("<") - 1);
			String[] nexCons = nexCon.split(" ");
			int lenght = nexCons.length;
			
			for (int i = 0; i < sentenceNum; i++){
				//String line = myText[i];
				List<CoreLabel> tokens = textList.get(i);
				int tokenSize = tokens.size();
				
				for (int j = 0; j < tokenSize - 1; j++){
					boolean flag = false;
					if (tokens.get(j).get(LemmaAnnotation.class).equalsIgnoreCase(nexCons[0]) 
							&&  tokenSize - j > lenght){
						flag  = true;
						for (int k = 0; k < lenght; k++){
							//String tempNex = tokens.get(j+k).get(TextAnnotation.class);
							if (!tokens.get(j+k).get(LemmaAnnotation.class).equalsIgnoreCase(nexCons[k])){
								flag = false;
								break;
							}
						}
					}
					
					if (flag == true){
						int end = j + lenght - 1;
						//the last token cannot be upperCase.
						String lastToken = tokens.get(end).get(TextAnnotation.class);
						if(lastToken.equals(lastToken.toLowerCase())){
							if(!inTargetEntity(tokens, j, end)){
								String targStr = getNexTarget(tokens, end);
								targStr = targStr.trim();
								if (!targStr.equals("") && targStr.length() >= 3){
									String[] preTerm = targStr.split(" ");
									if(preTerm.length > 1){
										//targStr is the extracted entity
										targStr = clearNextPosEntity(targStr, patStr, tokens, end, lenght);
										//extracted entity is added to patternMap's unlabeled entity arrayList
										if(!targStr.equals("")){
											Pattern pat = patternMap.get(patStr);
											if(!learnedWordList.contains(targStr)){
												pat.addExtWords(targStr);
											}else{
												pat.addPosWords(targStr);
											}
											patternMap.put(patStr, pat);
											
										}
										
									}else{
										//List<CoreLabel> targetTokens =getTokens(targStr);
										String targetPos = tokens.get(end + 1).get(PartOfSpeechAnnotation.class);
												//targetTokens.get(0).get(PartOfSpeechAnnotation.class);
										if(!targetPos.startsWith("N")||isTarget(commWords, targStr.toLowerCase())){
											targStr = clearNegEntity(targStr);
											//extracted entity is added to patternMap's negative entity arrayList
											Pattern pat = patternMap.get(patStr);
											pat.addNegWords(targStr);
											patternMap.put(patStr, pat);
										}else{
											targStr = clearNextPosEntity(targStr, patStr, tokens, end, lenght);
											//extracted entity is added to patternMap's unlabeled entity arrayList
											if(!targStr.equals("")){
												Pattern pat = patternMap.get(patStr);
												if(!learnedWordList.contains(targStr)){
													pat.addExtWords(targStr);
												}else{
													pat.addPosWords(targStr);
												}
												patternMap.put(patStr, pat);
												
											}
										}
									}
									
									//printOutTest(line, patStr, targStr);	
								}
								j = j + lenght - 1;
							}
						}
					
						
					}
				}
				
			}
		}
		
		private static String removeLastLowCase(String targStr){
			String result = "";
			List<CoreLabel> tokens = getTokens(targStr);
			int size = tokens.size();
		    while(size >= 0){
		    	String lastStr = tokens.get(size - 1).get(TextAnnotation.class);
				String lastPos = tokens.get(size - 1).get(PartOfSpeechAnnotation.class);
				if(lastStr.equals(lastStr.toLowerCase()) && !isVersionStr(lastStr, lastPos)){
					size = size - 1;
				}else{
					break;
				}
		    }
		    
		    for(int i = 0; i < size; i++){
		    	result += tokens.get(i).get(TextAnnotation.class) + " ";
		    }
		    
		    result = result.trim();
			return result;
		}
		
		private static String getNexTarget(List<CoreLabel> tokens, int start){
			// start<= tokens.size() - 2;
			String result = "";
			int nexBreak = getNextPoint(tokens, start, tokens.size());
			int length = nexBreak - start - 1;
			
			if (length > 0){
				if (length == 1){
					String str = tokens.get(start + 1).get(TextAnnotation.class);
					//String pos = tokens.get(start + 1).get(PartOfSpeechAnnotation.class);
					result = str;
					
				}
				
				if (length >= 2){
					String first = tokens.get(start + 1).get(TextAnnotation.class);
					String second = tokens.get(start + 2).get(TextAnnotation.class);
					String secondPos = tokens.get(start + 2).get(PartOfSpeechAnnotation.class);
					if (!first.equals(first.toLowerCase())){
						result += first + " ";
						for (int i = start +2; i < nexBreak; i++){
							String temp = tokens.get(i).get(TextAnnotation.class);
							String pos = tokens.get(i).get(PartOfSpeechAnnotation.class);
							if (!temp.equals(temp.toLowerCase())){
								result += temp + " ";
							}else{
								if(isValidWord(temp, pos)){
									result += temp + " ";
								}else{
									break;
								}
							}
						}
						result = removeLastLowCase(result.trim());
						
					}else{
						if(isVersionStr(second, secondPos)){
							result = first + " " + second;
						}else{
							result = first;
						}
						
					}
					
				}
				
			}
			
		/*	if(result.length() < 3){
				result = "";
			}*/
			
			return result;
			
		}
		
	
	
		
		private static void putExtWordScore(String extEntity, String pattern, double hasUpCase, 
				double hasNumber, double leftConHasTrig, double rightConHasTrig){
			if(!entityMap.containsKey(extEntity)){
				Entity newEnt = new Entity(extEntity);
				newEnt.setUpCase(hasUpCase);
				newEnt.setVersionNumber(hasNumber);
				newEnt.setLeftTrigger(leftConHasTrig);
				newEnt.setRightTrigger(rightConHasTrig);
				newEnt.addPatterns(pattern);
				entityMap.put(extEntity, newEnt);
				
			}else{
				Entity oldEntity = entityMap.get(extEntity);
				if(hasUpCase == 1 && oldEntity.getUpCase() != 1){
					oldEntity.setUpCase(1);
				}
				if(hasNumber == 1 && oldEntity.getVersionNumber() != 1){
					oldEntity.setVersionNumber(1);
				}
				if(leftConHasTrig == 1 && oldEntity.getLeftTrigger() != 1){
					oldEntity.setLeftTrigger(1);
				}
				if(rightConHasTrig == 1 && oldEntity.getRightTrigger() != 1){
					oldEntity.setRightTrigger(1);
				}
				if(leftConHasTrig == -0.4 && oldEntity.getLeftTrigger() == 0.2){
					oldEntity.setLeftTrigger(-0.4);
				}
				if(rightConHasTrig == -0.4 && oldEntity.getRightTrigger() == 0.2){
					oldEntity.setRightTrigger(-0.4);
				}
				if(!oldEntity.containPattern(pattern)){
					oldEntity.addPatterns(pattern);
				}
				
				entityMap.put(extEntity, oldEntity);
				
			}
	
		}
		
		private static String clearPosEntity(String targStr, String pattern, 
				List<CoreLabel> originTokens, int start){
			//targStr is the extracted entity, contFirst is the first word of the pattern, 
			//start is the begin index of pattern
			String result = "";
			
				List<CoreLabel> targetTerm = getTokens(targStr);
				if(targetTerm.size() <= 5){
					
				//start is the begin index of pattern
					String conFirst = originTokens.get(start).get(LemmaAnnotation.class);
					String conSecond = originTokens.get(start+1).get(LemmaAnnotation.class);
					String lastTerm = targetTerm.get(targetTerm.size()-1).get(LemmaAnnotation.class);
					/*if(isTarget(deletTarget, lastTerm)){
						result = clearNegEntity(targStr);
						Pattern pat = patternMap.get(pattern);
						pat.addNegWords(result);
						patternMap.put(pattern, pat);
						result = "";
						
					}else{*/
							
							//extracted entity has upperCase letter? 1 means have upperCase letter; 0.5 means no.
							double hasUpCase = 0.6;
							if(!targStr.equals(targStr.toLowerCase())){
								hasUpCase = 1.0;
							}
							//extracted entity has version number? 1 means have version number; 0.5 means no
							double hasNumber = 0.2;
							
							// result is the formated extracted entity
							result = formatNewEntity(targStr, targetTerm);
							for (int i = targetTerm.size() - 1; i >= 0; i--){
								String str = targetTerm.get(i).get(TextAnnotation.class);
								String pos = targetTerm.get(i).get(PartOfSpeechAnnotation.class);
								if(isVersionStr(str, pos)){
									hasNumber = 1.0;
									break;
								}
							}
							
							//context of extracted entity has extracting trigger words, such as software, package, and tool which is in triggerwords.txt
							// or has deleting trigger words, such as function, assay, and database which is target.txt?
							// 1 means has extracting trigger words, 0 means has deleting trigger words, 0.5 means has none of both
							double leftConHasTrig = 0.2;
							double rightConHasTrig = 0.2;
							if(isTarget(triggerWords, lastTerm) || isTarget(triggerWords, conFirst)){
								rightConHasTrig = 1.0;
							}
							
							if(isTarget(deletTarget, conFirst) || isTarget(deletTarget, conSecond)||isTarget(deletTarget, lastTerm)){
								rightConHasTrig = -0.4;
							}
							
							 if(start - targetTerm.size() - 1 >= 0){
									String preString = originTokens.get(start - targetTerm.size() - 1).get(LemmaAnnotation.class);
								   if(start - targetTerm.size() - 1 == 0){
									   if(isTarget(triggerWords, preString)){
									    	leftConHasTrig = 1.0;
										}
										if(isTarget(deletTarget, preString)){
											leftConHasTrig = -0.4;
										} 
								   }else{
									   String preSecString = originTokens.get(start - targetTerm.size() - 2).get(LemmaAnnotation.class);
										if(isTarget(triggerWords, preString)){
											leftConHasTrig = 1.0;
										}
										if(isTarget(deletTarget, preString)||isTarget(deletTarget, preSecString)){
											leftConHasTrig = -0.4;
										} 
								   }
									
								}
							
							
								int index = result.indexOf(" ");
								if(index < 0){
									//String myPos = originTokens.get(start - 1).get(PartOfSpeechAnnotation.class);
									String myPos = originTokens.get(start - targetTerm.size()).get(PartOfSpeechAnnotation.class);
									if(isTarget(commWords, result) || !myPos.startsWith("N")|| 
											isSingleLowcase(result, leftConHasTrig, rightConHasTrig)){
										
										result = clearNegEntity(result);
										Pattern pat = patternMap.get(pattern);
										pat.addNegWords(result);
										patternMap.put(pattern, pat);
										result = "";
										
									}else{
										putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
										
									}
										
								}else{
									putExtWordScore(result, pattern, hasUpCase, hasNumber, leftConHasTrig, rightConHasTrig);
								}
						
					//}
					
				}	
			return result;
		}
		
		private static String clearNegEntity(String target){
			boolean flag = true;
			
			//String result = "";
			String result = target;
			for (int i = 0; i < negativeEntity.size(); i++){
				String myEnt = negativeEntity.get(i);
				if (myEnt.equalsIgnoreCase(target)){
					result = myEnt;
					flag = false;
					break;
				}
				
			}
			
			if (flag == true){
				result = target;
				negativeEntity.add(result);
			}
			
			return result;
		}
		
		private static void extraPreWords(String patStr){
			//example of patStr: <> software package
			String preCon = patStr.substring(patStr.indexOf(">") + 2);
			String[] preCons = preCon.split(" ");
			int length = preCons.length;
			
			for (int i = 0; i < sentenceNum; i++){
				//String line = myText[i];
				List<CoreLabel> tokens = textList.get(i);
				int tokenSize = tokens.size();
				for (int j = 1; j < tokenSize; j++){
					boolean flag = false;
					String temp = tokens.get(j).get(TextAnnotation.class);
					if (temp.equals(temp.toLowerCase()) && 
							tokens.get(j).get(LemmaAnnotation.class).equalsIgnoreCase(preCons[0]) 
							&& tokenSize - j >= length){
						flag = true;
						for (int k = 0; k < length; k++){
							if ((!tokens.get(j + k).get(LemmaAnnotation.class).equalsIgnoreCase(preCons[k]))){
								flag = false;
								break;
							}
						}
					}
					
					if(flag == true){
						int end = j + length - 1;
						if(!inTargetEntity(tokens, j, end)){
							String targStr = getPreTarget(tokens, j);
							targStr = targStr.trim();
							if(!targStr.equals("") && targStr.length() >= 3){
								String[] preTerm = targStr.trim().split(" ");
								if(preTerm.length > 1){
									//targStr is the extracted entity, 
									targStr = clearPosEntity(targStr, patStr, tokens, j);
									//extracted entity is added to patternMap's unlabeled entity arrayList
									if(!targStr.equals("")){
										Pattern pat = patternMap.get(patStr);
										if(!learnedWordList.contains(targStr)){
											pat.addExtWords(targStr);
										}else{
											pat.addPosWords(targStr);
										}
										patternMap.put(patStr, pat);
										
									}
								}else{
									//List<CoreLabel> targetTokens =getTokens(targStr);
									String targetPos =  tokens.get(j-1).get(PartOfSpeechAnnotation.class);
									if(!targetPos.startsWith("N")||isTarget(commWords, targStr.toLowerCase())){
										targStr = clearNegEntity(targStr);
										//extracted entity is added to patternMap's negative entity arrayList
										Pattern pat = patternMap.get(patStr);
										pat.addNegWords(targStr);
										patternMap.put(patStr, pat);
									}else{
										targStr =  clearPosEntity(targStr, patStr, tokens, j);
										//extracted entity is added to patternMap's unlabeled entity arrayList
										if(!targStr.equals("")){
											Pattern pat = patternMap.get(patStr);
											if(!learnedWordList.contains(targStr)){
												pat.addExtWords(targStr);
											}else{
												pat.addPosWords(targStr);
											}
											patternMap.put(patStr, pat);
											
										}
										
									}
								}
								
								//printOutTest(line, patStr, targStr);
							}
							
							j = j + length - 1;
							
						}
						
					}
				}
			}
		}
		
		private static String removeStartLowCase(String text){
			String result = ""; 
			String[] tokens = text.split(" ");
			int size = tokens.length;
			
				int j = 0;
				for(int i = 0; i < size; i++){
					String temp = tokens[i];
					if(!temp.equals(temp.toLowerCase())){
						j = i;
						break;
					}
				}
				
				if(j > 0){
					for(int i = j; i < size; i++){
						result += tokens[i] + " ";
					}
					result = result.trim();
				}else{
					result = text;
				}
			
			
			return result;
			
			
		}
		private static boolean isValidWord(String secondStr, String secondPos){
			secondStr = secondStr.toLowerCase();
			boolean flag = false;
			if(secondStr.equals("and") || secondStr.equals("for") || 
					secondStr.equals("of") || secondStr.equals("or")
					||isVersionStr(secondStr, secondPos)){
				flag = true;
			}
			return flag;
		}
		
		private static String getPreTarget(List<CoreLabel> tokens, int start){
			//start >= 1;
			String result = "";
			int preBreak =  getPrePoint(tokens, start);
			int length = start - preBreak - 1;
			if (length > 0){
				if (length == 1){
					String str = tokens.get(start-1).get(TextAnnotation.class);
					//String pos = tokens.get(start - 1).get(PartOfSpeechAnnotation.class);
					result = str;

				}
				
				if (length >=2){
					String firstStr = tokens.get(start-1).get(TextAnnotation.class);
					String pos = tokens.get(start-1).get(PartOfSpeechAnnotation.class);
					
					String secd = tokens.get(start-2).get(TextAnnotation.class);
					if (!firstStr.equals(firstStr.toLowerCase())){
						result = firstStr;
						for(int i = start - 2; i > preBreak; i--){
							String secondStr = tokens.get(i).get(TextAnnotation.class);
							String secondPos = tokens.get(i).get(PartOfSpeechAnnotation.class);
							if (!secondStr.equals(secondStr.toLowerCase())){
								result = secondStr + " " + result;
							}else{
								//
								if (isValidWord(secondStr, secondPos) && i - 1 > preBreak){
									String preStr = tokens.get(i - 1).get(TextAnnotation.class);
									String prePos = tokens.get(i - 1).get(PartOfSpeechAnnotation.class);
									if (!preStr.equals(preStr.toLowerCase())||isVersionStr(preStr, prePos)){
										result = secondStr + " " + result;
									}else{
										break;
									}
									
								}else{
									break;
								}
							}
						}
						
						result = removeStartLowCase(result.trim());
						
					}else{
						if(isVersionStr(firstStr, pos)){
							if(!secd.equals(secd.toLowerCase())){
								result = secd + " " + firstStr;
								for(int j = start - 3; j > preBreak; j--){
									String temp = tokens.get(j).get(TextAnnotation.class);
									String tempPro = tokens.get(j).get(PartOfSpeechAnnotation.class);
									if (!temp.equals(temp.toLowerCase())){
										result = temp + " " + result;
									}else{
										//
										if(isValidWord(temp, tempPro) && j - 1 > preBreak){
											String preStr = tokens.get(j - 1).get(TextAnnotation.class);
											String prePos = tokens.get(j - 1).get(PartOfSpeechAnnotation.class);
											if (!preStr.equals(preStr.toLowerCase()) || isVersionStr(preStr, prePos)){
												result = temp + " " + result;
											}else{
												break;
											}
											
										}else{
											break;
										}
									}
								}
								
								result = removeStartLowCase(result);
								
							}else{
								result = secd + " " + firstStr;
								
							}
						}else{
							result = firstStr;
							
						}
						
					}
					
					
				}
			}
			
			/*if(result.length() < 3){
				result = "";
			}*/
			return result;
			
		}
		
		
		
		
		private static String formatNewEntity(String target, List<CoreLabel> targetTerm){
			String result = "";
		    String myresult = target;
			String temp = "";
			String lastStr = targetTerm.get(targetTerm.size() - 1).get(TextAnnotation.class);
			String lastPos = targetTerm.get(targetTerm.size() - 1).get(PartOfSpeechAnnotation.class);
			if(isVersionStr(lastStr, lastPos)){
				for(int i = 0; i < targetTerm.size() - 1; i++){
					temp += targetTerm.get(i).get(TextAnnotation.class) + " ";
				}
				myresult = temp.trim();
			}
			
			boolean flag = true;
			for (int i = 0; i < allEntities.size(); i++){
				String myEnt = allEntities.get(i);
				if (myEnt.equalsIgnoreCase(myresult)){
					result = myEnt;
					flag = false;
					break;
				}
				
			}
			
			if (flag == true){
				result = myresult;
				allEntities.add(myresult);
			}
			
			return result;
			
		}

		
		

		private static void removeStopWords(){
			//read in stop words
			String stopWordsFile = inDir + "/punctuation.txt";
			Database db = new Database(stopWordsFile);
			String[] set = new String[db.getSize()];
			
			Iterator<String> stopIt = db.getIterator();
			for (int j = 0; stopIt.hasNext(); j++){
				set[j] = stopIt.next();
			}
			
			for (int k = 0; k < myText.length; k++){
				String text = myText[k];
				String result = "";
				if (!text.trim().equals("")){
					text = removeLRB(text, "(", ")");
					text = removeLRB(text, "[", "]");
					text = removeLRB(text, "“", "”");
					text = removeLRB(text, "‘", "’");
					if(!text.equals("")){
						List<CoreLabel> tokens = getTokens(text);
						for (int i = 0; i< tokens.size(); i++){
							String myToken = tokens.get(i).get(TextAnnotation.class);
							
							if (Arrays.binarySearch(set, myToken.toLowerCase()) >= 0){
								myToken = "";
							}
							if((myToken.startsWith("v") || myToken.startsWith("V")) && myToken.length() > 1){
								if(Character.isDigit(myToken.charAt(1))){
									myToken = "";
								}
							}
							if (!myToken.equals("")){
								result += myToken + " ";
							}
						}
						//result = removeBracket(result.trim());
						result = result.trim();
						
					}
						
					
					
				}
				myText[k] = result;
				
			}
			
		}
		private static String removeLRB(String line, String leftStr, String rightStr){
			String result = "";
			int start =  line.indexOf(leftStr);
			if (start == -1) result = line;
			int end = -1;
			int secondSt = -1;
			int secondEn = -1;
			while(start != -1 ){
				end = line.indexOf(rightStr);
				if (end != -1){
					secondSt = line.substring(start + 1).indexOf(leftStr);
					secondEn = line.substring(start + 1).indexOf(rightStr);
					if (secondSt == -1){
						result += line.substring(0, start) + line.substring(end+1);
						break;
					}
					if (secondSt != -1){
						if (secondSt > secondEn){
							result += line.substring(0, start);
							line = line.substring(end + 1);
							start = line.indexOf(leftStr);
						}else{
							result = line;
							break;
						}
					}
				}else{
					result = line;
					break;
				}
				
			}
			return result;
			
		}
		
		private static boolean contextHasTriggers(String pattStr, String[] myTargets){
			boolean flag = false;
			String[] temps = pattStr.split(" ");
			for(int i = 0; i < temps.length; i++){
				String str = temps[i];
				if(isTarget(myTargets, str)){
					flag = true;
					break;
				}
			}
			
			return flag;
		}

		
		private static void getCandidatePattern(int n, String[] seeds){
			//String myStr
	            ArrayList<String> context = new ArrayList<String>();
	            List<CoreLabel> tokens = textList.get(n);
	            String target = "";
				int index = getIndex(tokens, "<software>");
				while (index >= 0){
					 context = new ArrayList<String>();
					 target = "";
					 int start = getIndex(tokens, "<software>");
					 int end = getIndex(tokens, "</software>");
					 
					 for (int j = start + 1; j < end; j++){
						 target += tokens.get(j).get(TextAnnotation.class) + " ";
					 }
					 
					 target = target.trim();
					 for(int i = 0; i < seeds.length; i++){
						 String seed = seeds[i];
						 if(target.equalsIgnoreCase(seed)){
							 context = getContext(tokens, start, end);
						 }
						 
					 }
					 
					 if(context.size()>0){
						 // add target and its context to patternMap
						 for (int k = 0; k < context.size(); k++){
							 String pattStr = context.get(k);
							 //printTest(pattStr);
							 if (!patternMap.containsKey(pattStr)){
								 Pattern newPattern = new Pattern(pattStr);
								 newPattern.addPosWords(target);
								 if(contextHasTriggers(pattStr, contextTriggers)){
									 newPattern.setTrigWordFlag(true);
								 }
								 patternMap.put(pattStr, newPattern);
								
							 }else{
								 Pattern oldPattern = patternMap.get(pattStr);
								 oldPattern.addPosWords(target);
								 patternMap.put(pattStr, oldPattern);
							 }
							 
							 
						 }
					 }
					 
					 if (end < tokens.size() -1){
						 tokens = tokens.subList(end + 1, tokens.size());
						 index = getIndex(tokens, "<software>");
						 
					 }else{
						 break;
					 }
					 
					
				}
				
			
				
				
		}
		
		private static ArrayList<String> getContext(List<CoreLabel> tokens, int start, int end){
			
			ArrayList<String> context = new ArrayList<String>();
			ArrayList<String> preContext = new ArrayList<String>();
			ArrayList<String> nexContext = new ArrayList<String>();
			boolean preFlag = false;
			boolean nextFlag = false;
			//get 2-4 tokens before the target
			if (start > 0){
				//int prePoint = getPrePoint(tokens, start);
				//the token previous of target must be low case
				String preToken = tokens.get(start - 1).get(TextAnnotation.class);
				String preTokenPos = tokens.get(start - 1).get(PartOfSpeechAnnotation.class);
				String preLemma = tokens.get(start - 1).get(LemmaAnnotation.class);
				//if(preToken.equals(preToken.toLowerCase()) && !isTarget(deletTarget, preLemma))
				if(preToken.equals(preToken.toLowerCase()) && Character.isLetter(preToken.charAt(0))){
					int preBreak = 4;
					if (start < 4){
						preBreak = start;
					}
					boolean[] stopFlag = new boolean[preBreak];
					String result = "";
					int numOfStopWord = 0;
					for (int i = 1; i <= preBreak; i++){
						String myPreText = tokens.get(start - i).get(TextAnnotation.class).toLowerCase();
						String myPrePos = tokens.get(start - i).get(PartOfSpeechAnnotation.class);
						if(isTarget(stopWords, myPreText) || myPrePos.equals("CD")){
							stopFlag[preBreak - i] = true;
							numOfStopWord++;
						}
						result = tokens.get(start - i).get(LemmaAnnotation.class) + " " + result;
						
						if (!result.trim().equals("") && !preContext.contains(result.trim())){
							preContext.add(result.trim());
						}
					
					}
					
					
					//discard patterns whose left or right context was 1 or 2 stop words to 
					//avoid generating low precision patterns; Three or more stop words resulted in some
					//patterns.
					preContext = clearPatterns(preContext, stopFlag, numOfStopWord);
					//add mid context
					if(!isTarget(stopWords, preToken.toLowerCase()) && !preTokenPos.equals("CD")){
						preContext.add(tokens.get(start-1).get(LemmaAnnotation.class));
					}
					
					if(isTarget(triggerWords, preToken.toLowerCase())){
						preFlag = true;
					}
					
				}
				
			}
				
			//get context after the target
			end = end + 1;
			int size = tokens.size();
			if (end < size){
				String afterToken = tokens.get(end).get(TextAnnotation.class);
				String afterTokenPos = tokens.get(end).get(PartOfSpeechAnnotation.class);
				String afterLemma = tokens.get(end).get(LemmaAnnotation.class);
				//if(afterToken.equals(afterToken.toLowerCase()) && !isTarget(deletTarget, afterLemma))
				if((afterToken.equals(afterToken.toLowerCase()) || 
						afterToken.equals("Software") ) &&
						Character.isLetter(afterToken.charAt(0))){
					int nextBreak = 4;
					if (size - end < 4){
						nextBreak = size - end;
					}
					boolean[] stopFlag = new boolean[nextBreak];
					String result = "";
					int numOfStopWord = 0;
					for(int i = 1; i <= nextBreak; i++){
						String myNexText = tokens.get(end - 1 + i).get(TextAnnotation.class).toLowerCase();
						String myNexPos = tokens.get(end - 1 + i).get(PartOfSpeechAnnotation.class);
						if(isTarget(stopWords, myNexText) || myNexPos.equals("CD")){
							stopFlag[i-1] = true;
							numOfStopWord++;
						}
						String myNex = tokens.get(end - 1 + i).get(LemmaAnnotation.class);
						result += myNex + " ";
						
						if ((!result.trim().equals("")) && !nexContext.contains(result.trim())){
							nexContext.add(result.trim());
						}
					}
					//discard patterns whose left or right context was 1 or 2 stop words to 
					//avoid generating low precision patterns; Three or more stop words resulted in some
					//patterns.
					nexContext = clearPatternsNext(nexContext, stopFlag, numOfStopWord);
					//add mid context
					
					if(!isTarget(stopWords, afterToken.toLowerCase()) && !afterTokenPos.equals("CD")){
						nexContext.add(tokens.get(end).get(LemmaAnnotation.class));
					}
					if(isTarget(triggerWords, afterToken.toLowerCase())){
						nextFlag = true;
					}
					
				
			   }
			}
					
			
			//add mid context
			String lastPrevious = "";
			String lastNext = "";
			if(preContext.size() >= 1){
				int mysize = preContext.size();
				String temp = preContext.get(mysize- 1);
				int index = temp.indexOf(" ");
				if(index < 0){
					mysize = mysize - 1;
					if(preFlag == true || nextFlag == true){
						lastPrevious = temp;
					}
					
				}
				for(int i = 0; i < mysize; i++){
					context.add(preContext.get(i) + " <>");
				}
			}
			
			if(nexContext.size() >=1){
				int nexSize = nexContext.size();
				String nexTemp = nexContext.get(nexSize - 1);
				int nexIndex = nexTemp.indexOf(" ");
				if(nexIndex < 0){
					nexSize = nexSize - 1;
					if(preFlag == true || nextFlag == true){
						lastNext = nexTemp;
					}
					
				}
				for(int i = 0; i < nexSize; i++){
					context.add("<> " + nexContext.get(i));
				}
			}
			if(!lastPrevious.equals("") && !lastNext.equals("")){
				context.add(lastPrevious + " <> " + lastNext);
			}
			
			
			
			return context;
			
			
			
		}
		
		private static ArrayList<String> clearPatternsNext(ArrayList<String> nexContext, 
				boolean[] stopFlag, int num){
			// num denotes the number of stop words
			ArrayList<String> saveContext = new ArrayList<String>();
			
			int length = stopFlag.length;
			
			if(length == 2){
				if (num == 0){
					saveContext.add(nexContext.get(1));
				}
			}
			
			if(length == 3){
				if(num == 0){
					saveContext.add(nexContext.get(1));
					saveContext.add(nexContext.get(2));
				}
				
				if (num == 1){
					if (stopFlag[2] == false){
						saveContext.add(nexContext.get(2));
						
					}else{
						saveContext.add(nexContext.get(1));
						saveContext.add(nexContext.get(2));
					}
					
				}
			}
			
			if (length == 4){
				if(num == 0){
					saveContext.add(nexContext.get(1));
					saveContext.add(nexContext.get(2));
					saveContext.add(nexContext.get(3));
				}
				if(num == 1){
					if (stopFlag[0] == true || stopFlag[1] == true){
						saveContext.add(nexContext.get(2));
						saveContext.add(nexContext.get(3));
					}
					if(stopFlag[2] == true || stopFlag[3] == true){
						saveContext.add(nexContext.get(1));
						saveContext.add(nexContext.get(2));
						saveContext.add(nexContext.get(3));
					}
					
				}
				if(num == 2){
					if (stopFlag[3] == false){
						saveContext.add(nexContext.get(3));
					}else{
						if(stopFlag[2] == true){
							saveContext.add(nexContext.get(1));
							saveContext.add(nexContext.get(2));
							saveContext.add(nexContext.get(3));
						}else{
							saveContext.add(nexContext.get(2));
							saveContext.add(nexContext.get(3));
						}
					}
					
				}
				
				if (num == 3){
					saveContext.add(nexContext.get(3));	
				}
				
				if (num == 4){
					saveContext.add(nexContext.get(3));
				}
			}
			//saveContext = processPatterns(saveContext);
			
			return saveContext;
		
		}
		
		private static ArrayList<String> clearPatterns(ArrayList<String> preContext, 
				boolean[] stopFlag, int num){
			ArrayList<String> saveContext = new ArrayList<String>();
			
			int length = stopFlag.length;
			
			if(length == 2){
				if (num == 0){
					saveContext.add(preContext.get(1));
				}
			}
			
			if(length == 3){
				if(num == 0){
					saveContext.add(preContext.get(1));
					saveContext.add(preContext.get(2));
				}
				if (num == 1){
					if (stopFlag[0] == false){
						saveContext.add(preContext.get(2));
					}else{
						saveContext.add(preContext.get(1));
						saveContext.add(preContext.get(2));
					}
					
				}
				
			}
			
			if (length == 4){
				if(num == 0){
					saveContext.add(preContext.get(1));
					saveContext.add(preContext.get(2));
					saveContext.add(preContext.get(3));
				}
				if(num == 1){
					if (stopFlag[3] == true || stopFlag[2] == true){
						saveContext.add(preContext.get(2));
						saveContext.add(preContext.get(3));
					}
					if(stopFlag[1] == true || stopFlag[0] == true){
						saveContext.add(preContext.get(1));
						saveContext.add(preContext.get(2));
						saveContext.add(preContext.get(3));
					}
					
				}
				if(num == 2){
					if (stopFlag[0] == false){
						saveContext.add(preContext.get(3));
					}else{
						if(stopFlag[1] == true){
							saveContext.add(preContext.get(1));
							saveContext.add(preContext.get(2));
							saveContext.add(preContext.get(3));
						}else{
							saveContext.add(preContext.get(2));
							saveContext.add(preContext.get(3));
						}
					}
					
				}
				
				if (num == 3){
					
					saveContext.add(preContext.get(3));
				}
				
				if (num == 4){
					saveContext.add(preContext.get(3));
				}
			}
			
			//saveContext = processPatterns(saveContext);
			
			return saveContext;
		
			
		}
		
		
		private static boolean isTarget(String[] targets, String str){
			boolean flag = false;
			int index = Arrays.binarySearch(targets, str.toLowerCase());
			if (index < 0){
				flag = false;
			}else{
				flag = true;
			}
			
			return flag;
		}
		
		private static int getPrePoint(List<CoreLabel> tokens, int start){
			int breakPoint = -1;
			for (int i = start - 1; i >= 0; i--){
				String word = tokens.get(i).get(TextAnnotation.class);
				String wordPos = tokens.get(i).get(PartOfSpeechAnnotation.class);
				if(!isAlphanumberic(word)){
					breakPoint = i;
					break;
				}
				if(wordPos.startsWith("V")){
					breakPoint = i;
					break;
				}
				
			}
			
			return breakPoint;
		}
		
		private static int getNextPoint(List<CoreLabel> tokens, int end, int size){
			// size is the tokens size
			int breakPoint = size;
			for (int i = end + 1; i <= size - 1; i++){
				String word = tokens.get(i).get(TextAnnotation.class);
				String wordPos = tokens.get(i).get(PartOfSpeechAnnotation.class);
				if (!isAlphanumberic(word)){
					breakPoint = i;
					break;
				}
				if(wordPos.startsWith("V")){
					breakPoint = i;
					break;
				}
			}
			
			return breakPoint;
		}
		
		private static boolean isAlphanumberic(String word){
			for (int i = 0; i < word.length(); i++){
				char c = word.charAt(i);
				if ((c == '-'||c == '/'||c == '.') && i < word.length() - 1){
					i++;
					c = word.charAt(i);
				}
				
				if (!Character.isLetterOrDigit(c))
					return false;
			}
			return true;
			
		}
		
		private static boolean isVersionStr(String temp, String pos){
			boolean flag = false;
			if(pos.equals("CD") && !Character.isLetter(temp.charAt(0))){
				flag = true;
			}
			if(temp.length() >= 2){
				if((temp.startsWith(".") && Character.isDigit(temp.charAt(1)))){
					flag = true;
				}
			}
			return flag;
		}
		
		private static int getIndex(List<CoreLabel> tokens, String myMark){
			for (int i = 0; i < tokens.size(); i++){
				if(tokens.get(i).get(TextAnnotation.class).equals(myMark))
					return i;
			}
			return -1;
		}
		
		
		private static String getLabeledText(String text, String seed){
			 List<CoreLabel> tokens = getTokens(text);
				 List<CoreLabel> result = markedText(tokens, seed);
				 StringBuffer ss = new StringBuffer();
				 for (int i = 0; i < result.size(); i++){
					 String temp = result.get(i).get(TextAnnotation.class);
					 if (temp.equals("</software>") && i < result.size() - 1){
						 String secondStr = result.get(i + 1).get(TextAnnotation.class);
						 String secondPos = result.get(i + 1).get(PartOfSpeechAnnotation.class);
						 if(isVersionStr(secondStr, secondPos)){
							 i++;
						 }
					 }
					 ss.append(temp).append(" ");
					
				 }
				 
				 String total = ss.toString().trim();
				 
			 
			 return total;
		}
		private static void labelSeed(int index, String seed){
	
			String text = myText[index];
			int indexSoft = text.indexOf("<software>");
			if (indexSoft < 0){
				String temp = getLabeledText(text, seed);
				 myText[index] = temp;
				 textList.set(index,getTokens(temp));
			}else{
				String totalText = "";
				while (indexSoft >= 0){
					int endIndext = text.indexOf("</software>");
					String preText = text.substring(0, indexSoft);
					String midText = text.substring(indexSoft,  endIndext + 11);
					totalText += getLabeledText(preText, seed) + " " + midText + " ";
					text = text.substring(endIndext + 11).trim();
					indexSoft = text.indexOf("<software>");
					if (indexSoft < 0){
						totalText += getLabeledText(text, seed) + " ";
						break;
					}
				}
				myText[index] = totalText.trim();
				textList.set(index, getTokens(totalText.trim()));
				
			}
				
			
			
		}
		
		
		private static boolean neighberUpcase(List<CoreLabel> tokens, int index, int size){

			 boolean flagPre = false;
		        boolean flagNex = false;
		        if(index >= 1){
		            String preStr = tokens.get(index - 1).get(TextAnnotation.class);
		            String prePos = tokens.get(index - 1).get(PartOfSpeechAnnotation.class);
		           /* if(isTarget(deletTarget, preStr.toLowerCase())){
		            	flagPre = true;
		            }*/
		            if (!preStr.equals(preStr.toLowerCase()) && !prePos.startsWith("V")){
		                flagPre = true;
		            }
		            /*if(preStr.equalsIgnoreCase("and") || preStr.equalsIgnoreCase("or")){
		            	if(index > 2){
		            		flagPre = true;
		            	}
		            }*/
		          
		        }
		        
		        if (index + size < tokens.size()){
		            String nexStr = tokens.get(index + size).get(TextAnnotation.class);
		            String nexPos = tokens.get(index + size).get(PartOfSpeechAnnotation.class);
		           /* if(isTarget(deletTarget, nexStr.toLowerCase())){
		            	flagNex = true;
		            }*/
		           /* if (!nexStr.equals(nexStr.toLowerCase()) && !nexPos.startsWith("V")){
		                flagNex = true;
		            }*/
		            if (!nexStr.equals(nexStr.toLowerCase())  && !nexStr.equals("Software") ){
		                flagNex = true;
		            }
		        
		        }
		     
		        
		        return (flagPre||flagNex);
	    }
	    
	   private static String processOneWord(List<CoreLabel> tokens, String software){
			String result = "";
			int length = software.length();
			for (int i = 0; i < tokens.size(); i++){
				String target = tokens.get(i).get(TextAnnotation.class);
				if(length > 3){
					if (target.equalsIgnoreCase(software)){
						if(!neighberUpcase(tokens, i, 1)){
							if(!software.equals(software.toLowerCase())){
								if(!target.equals(target.toLowerCase())){
									//target = "<software> "  + target + " </software>";
									target = "<software> "  + software + " </software>";
								}
							}else{
								//target = "<software> "  + target + " </software>";
								target = "<software> "  + software + " </software>";
							}
							
						
						}
						
					}
				}else{
					if (target.equals(software)){
		                 if(!neighberUpcase(tokens, i, 1)){
		                      // target = "<software> "  + target + " </software>";
		                	 target = "<software> "  + software + " </software>";
		                  }
							
					}
				}
				
				result += target + " ";
			}
			result = result.trim();
			
			return result;
			
		}
		
		private static String processTwoWords(List<CoreLabel> tokens, String software){
			String result = "";
			String[] words = software.split(" ");
			String first = words[0];
			int size = words.length;
			for (int j = 0; j < tokens.size(); j++){
				String target = tokens.get(j).get(TextAnnotation.class);
				boolean flag = false;
				if (target.equalsIgnoreCase(first) && (tokens.size() - j) >= size ){
					flag = true;
					for (int k = 1; k < size; k++){
						if (!tokens.get(j+k).get(TextAnnotation.class).equalsIgnoreCase(words[k])){
							flag = false;
							break;
						}
							
					}
				}
				
				if (flag == true){
					if(!neighberUpcase(tokens, j, size)){
						String temp = "";
						for(int n = j; n < j + size; n++){
							temp += tokens.get(n).get(TextAnnotation.class) + " ";
						}
						temp = temp.trim();
						if(!software.equals(software.toLowerCase())){
							if (!temp.equals(temp.toLowerCase())){
								target = "<software> "  + temp + " </software>";
							}
						}else{
							target = "<software> "  + temp + " </software>";
						}
						//target = "<software> "  + temp + " </software>";
						
	                    j = j + size - 1;
					}
				}
				
				result += target + " ";
			}
			result = result.trim();
			return result;
		}
		
		private static List<CoreLabel> markedText(List<CoreLabel> tokens, String software){
		
			String[] words = software.split(" ");
			String result = "";
			// one seed only has one word
			if (words.length == 1){
				result = processOneWord(tokens, software);
			}	
				
			//one seed only has more than two words
			if (words.length > 1){
				result = processTwoWords(tokens, software);	
			}
					
		    tokens = getTokens(result);		
			
			return tokens;
			
		
			
		}
		
		private static List<CoreLabel> getTokens(String text){
			List<CoreLabel> tokens =new ArrayList<CoreLabel>();
			if (!text.trim().equals("")){
				List<CoreMap> sentences = getSentences(text);
				for(CoreMap sentence: sentences){
					for (CoreLabel token: sentence.get(TokensAnnotation.class))
						tokens.add(token);
				}
			    
			}
			return tokens;
			
		
	}

	private static List<CoreMap> getSentences(String text){
		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, pos, lemma, ner");
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		
		// create an empty Annotators on this text
		Annotation document = new Annotation(text);
		
		// run all Annotation on this text
		pipeline.annotate(document);
		
		// these are all the sentences in this document
		// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
		 List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		 return sentences;
	}
	
	private static boolean containSoftware(int n, String software){
		boolean flag = false;
		String[] words = software.split(" ");
		int length = software.length();
		int size = words.length;
		List<CoreLabel> tokens = textList.get(n);
		if(size == 1){
			for (int i = 0; i < tokens.size(); i++){
				String target = tokens.get(i).get(TextAnnotation.class);
				if(length > 3){
					if (target.equalsIgnoreCase(software)){
						if(!neighberUpcase(tokens, i, 1)){
							if(!software.equals(software.toLowerCase())){
								if(!target.equals(target.toLowerCase())){
									flag = true;
									break;
								}
							}else{
								flag = true;
								break;
							}
							
						}
						
					}
				}else{
					if (target.equals(software)){
		                 if(!neighberUpcase(tokens, i, 1)){
		                       flag = true;
		                       break;
		                  }
							
					}
				}
			}
		}else{
			String first = words[0];
			for (int j = 0; j < tokens.size(); j++){
				String target = tokens.get(j).get(TextAnnotation.class);
				flag = false;
				if (target.equalsIgnoreCase(first) && (tokens.size() - j) >= size ){
					flag = true;
					for (int k = 1; k < size; k++){
						if (!tokens.get(j+k).get(TextAnnotation.class).equalsIgnoreCase(words[k])){
							flag = false;
							break;
						}
							
					}
				}
			}
			
			
		}
		
		
		return flag;
	}
	
	/**
	 * label the text based on the words in the seed file
	 */
	private static void labelSoftware(String[] targets){
		lineTarget = new HashMap<Integer, Targets>();
		
		for (int i = 0; i < sentenceNum; i++){
			String myStr = myText[i];
			if (!myStr.trim().equals("")){
				for (int j = 0; j < targets.length; j++){
					String software = targets[j];
					if (software != null){
						if(containSoftware(i, software)){
							labelSeed(i, software);
							if(lineTarget.containsKey(i)){
								Targets targ = lineTarget.get(i);
								targ.addTargets(software);
								lineTarget.put(i, targ);
							}else{
								Targets targ = new Targets(i);
								targ.addTargets(software);
								lineTarget.put(i, targ);
							}
						}
						
					}
					
				}
				
			}
			
		}
		
	}

	

}













