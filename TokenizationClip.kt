package com.dynasoulstudio.jeremai

import android.util.Log
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.*

class TokenizationClip {
    val vocabularyFiles = mutableMapOf(
        "vocab_file" to "vocab.json",
        "merges_file" to "merges.txt",
    )
    val maxTokens = 77

    fun bytesToUnicode(): Map<Int, Char> {
        var bytes = (
                (33..126).toList() + (161..172).toList() + (174..255).toList()
                )
        var characters = bytes.toList()
        var number = 0
        for (b in 0..255) {
            if (b !in bytes) {
                bytes += b
                characters += (256 + number)
                number += 1
            }
        }
        val characterList = characters.map { it.toChar() }
        return bytes.zip(characterList).toMap()
    }

    fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        var prevChar = word[0]
        for (i in 1 until word.size) {
            val char = word[i]
            pairs.add(Pair(prevChar, char))
            prevChar = char
        }
        return pairs
    }


    fun whitespaceClean(text: String): String {
        val cleanedText = Regex("\\s+").replace(text, " ")
        return cleanedText.trim()
    }

    fun whitespaceTokenize(text: String): List<String> {
        val cleanedText = text.trim()
        return if (cleanedText.isNotEmpty()) {
            cleanedText.split(" ")
        } else {
            emptyList()
        }
    }

    inner class BasicTokenizer(doLowerCase: Boolean = true, neverSplit: Set<String>? = null, tokenizeChineseChars: Boolean = true, stripAccents: Boolean? = null) {

        private var doLowerCase: Boolean = true
        private var neverSplit: Set<String> = setOf()
        private var tokenizeChineseChars: Boolean = true
        private var stripAccents: Boolean? = null

        init {
            if (neverSplit == null) {
                this.neverSplit = setOf()
            }
            this.doLowerCase = doLowerCase
            if (neverSplit != null) {
                this.neverSplit = neverSplit
            }
            this.tokenizeChineseChars = tokenizeChineseChars
            this.stripAccents = stripAccents
        }
        fun tokenize(text: String, neverSplit: List<String>? = null): List<String> {
            val neverSplitSet = (if (neverSplit != null) this.neverSplit.union(neverSplit) else this.neverSplit).toSet()
            var text = _cleanText(text)
            if (this.tokenizeChineseChars) {
                text = this._tokenizeChineseChars(text)
            }
            val origTokens = whitespaceTokenize(text)
            val splitTokens = mutableListOf<String>()
            for (token in origTokens) {
                var workToken = token
                if (token !in neverSplitSet) {

                    if (this.doLowerCase) {
                        workToken = token.lowercase(Locale.ENGLISH)
                        if (stripAccents != false) {
                            workToken = this._runStripAccents(token)
                        }
                    } else if (this.stripAccents == true) {
                        workToken = this._runStripAccents(token)
                    }
                }
                splitTokens.addAll(_runSplitOnPunc(workToken, neverSplitSet))
            }
            val outputTokens = whitespaceTokenize(splitTokens.joinToString(separator = " "))
            return outputTokens
        }

        private fun _runStripAccents(text: String): String {
            return text.replace("[^A-Za-z0-9]".toRegex(), "")
        }

        private fun _runSplitOnPunc(text: String, neverSplit: Set<String>? = null): List<String> {
            if (neverSplit != null && text in neverSplit) {
                return listOf(text)
            }
            val chars = text.toCharArray().toList()
            var i = 0
            var startNewWord = true
            val output = mutableListOf<MutableList<Char>>()
            while (i < chars.size) {
                val char = chars[i]
                if (_isPunctuation(char)) {
                    output.add(mutableListOf(char))
                    startNewWord = true
                } else {
                    if (startNewWord) {
                        output.add(mutableListOf())
                    }
                    startNewWord = false
                    output[output.size - 1].add(char)
                }
                i += 1
            }
            return output.map { it.joinToString("") }
        }

        private fun _tokenizeChineseChars(text: String): String {
            val output = mutableListOf<Char>()
            for (char in text) {
                val cp = char.code
                if (_isChineseChar(cp)) {
                    output.add(' ')
                    output.add(char)
                    output.add(' ')
                } else {
                    output.add(char)
                }
            }
            return output.joinToString("")
        }

        private fun _isChineseChar(cp: Int): Boolean {
            return cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF || cp in 0x20000..0x2A6DF || cp in 0x2A700..0x2B73F || cp in 0x2B740..0x2B81F || cp in 0x2B820..0x2CEAF || cp in 0xF900..0xFAFF || cp in 0x2F800..0x2FA1F
        }

        private fun _cleanText(text: String): String {
            val output = mutableListOf<Char>()
            for (char in text) {
                val cp = char.code
                if (cp == 0 || cp == 0xFFFD || _isControl(char)) {
                    //Log.d("Tokenizer","Ignored character: "+char+" because: "+cp+" or: "+_isControl(char))
                    continue
                }
                if (_isWhitespace(char)) {
                    output.add(' ')
                } else {
                    output.add(char)
                }
            }
            return output.joinToString("")
        }

        private fun _isWhitespace(char: Char): Boolean {
            return char == ' ' || char == '\t' || char == '\n' || char == '\r' || Character.getType(char) == Character.SPACE_SEPARATOR.toInt()
        }

        private fun _isControl(char: Char): Boolean {
            if (char == '\t' || char == '\n' || char == '\r') {
                return false
            }
            val type = Character.getType(char)
            if (type == Character.CONTROL.toInt() || type == Character.FORMAT.toInt()) {
                return true
            }
            return false
        }




        private fun _isPunctuation(char: Char): Boolean {
            val cp = char.toInt()
            if (cp >= 33 && cp <= 47 || cp >= 58 && cp <= 64 || cp >= 91 && cp <= 96 || cp >= 123 && cp <= 126) {
                return true
            }
            return !char.isLetterOrDigit()
        }
    }

    inner class CLIPTokenizer(vocabFile: String,
                        mergesFile: String,
                        errors: String = "replace",
                        unkToken: String = "<|endoftext|>",
                        bosToken: String = "<|startoftext|>",
                        eosToken: String = "<|endoftext|>",
                        padToken: String = "<|endoftext|>"
    ) {
        private val vocabFilesNames = vocabularyFiles
        private val maxModelInputSizes = maxTokens
        private val modelInputNames = listOf("input_ids", "attention_mask")

        val errors: String = errors
        val unkToken: String = unkToken
        val bosToken: String = bosToken
        val eosToken: String = eosToken
        val padToken: String = padToken

        val nlp = BasicTokenizer(true)
        val fixText: Boolean = false

        val encoder: Map<String, Int>
        val decoder: Map<Any, String>
        val byteEncoder: Map<Int, Char>
        val byteDecoder: Map<Char, Int>
        val bpeRanks = mutableMapOf<Pair<String, String>, Int>()
        val cache = mutableMapOf("<|startoftext|>" to "<|startoftext|>", "<|endoftext|>" to "<|endoftext|>")
        val pat = Regex(
            """<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            RegexOption.IGNORE_CASE
        )


        init {
            val vocabHandle = File(vocabFile).inputStream().reader(Charsets.UTF_8).readText()
            encoder = Json.decodeFromString(vocabHandle)
            decoder = encoder.entries.associate { (v, k) -> k to v }
            byteEncoder = bytesToUnicode()
            byteDecoder = byteEncoder.entries.associate { (v, k) -> k to v }

            File(mergesFile).bufferedReader(Charsets.UTF_8).use { mergesHandle ->
                val bpeMerges = mergesHandle.readLines().map { it.trim() }.subList(1, 49152 - 256 - 2 + 1)
                bpeMerges.forEachIndexed { index, merge ->
                    val (a, b) = merge.split(" ")
                    bpeRanks[Pair(a, b)] = index
                }
            }
        }

        @get:JvmName("vocabSize")
        val vocabSize: Int
        get() = encoder.size


        fun bpe(token: String): String {
            if (token in cache) {
                return cache[token]!!
            }

            var word = listOf(
                token.substring(0, token.length - 1) +
                token.substring(token.length - 1) + "</w>"
            )
            var pairs = getPairs(word)

            if (pairs.isEmpty()) {
                return token + "</w>"
            }

            while (true) {
                val bigram =
                    pairs.minBy { pair -> bpeRanks.getOrDefault(pair, Int.MAX_VALUE) }
                if (!bpeRanks.contains(bigram)) {
                    break
                }
                val first = bigram.first
                val second = bigram.second
                val newWord = mutableListOf<String>()
                var i = 0
                while (i < word.size) {
                    var j = -1
                    for (k in i until word.size) {
                        if (word[k] == first) {
                            j = k
                            break
                        }
                    }

                    if (j == -1) {
                        newWord.addAll(word.subList(i, word.size))
                        break
                    } else {
                        newWord.addAll(word.subList(i, j))
                        i = j
                    }
                    if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                        newWord.add(first + second)
                        i += 2
                    } else {
                        newWord.add(word[i])
                        i += 1
                    }
                }
                word = newWord
                if (word.size == 1) {
                    break
                } else {
                    pairs = getPairs(word)
                }
            }

            val result = word.joinToString(" ")
            cache[token] = result
            return result
        }


        fun tokenize(text: String): IntArray {
            val bpeTokens = mutableListOf<String>()
            var tokenizedText: String
            if (fixText == null || fixText == false) {
                tokenizedText = nlp.tokenize(text).joinToString(" ")
            } else {
                tokenizedText = whitespaceClean(text).lowercase()
            }

            val tokens = Regex(pat.pattern).findAll(tokenizedText).map {
                it.value
            }


            for (token in tokens) {
                val byteEncodedToken = token.toByteArray(Charsets.UTF_8)
                    .map { byteEncoder[it.toInt()] }
                    .joinToString("")
                bpeTokens.addAll(bpe(byteEncodedToken).split(" "))
            }
            return buildInputsWithSpecialTokens(convertTokensToIds(bpeTokens) as List<Int>)
        }

        fun convertTokenToID(token: String): Int {
            var id = encoder.get(token)
            if(id != null) {
                return id
            }
            else{
                id = encoder.get(unkToken)
                return id!!
            }
        }

        fun convertIDToToken(id: Int): String {
            return decoder.get(id)!!
        }

        fun convertTokensToString(tokens: List<String>): String {
            val text = tokens.joinToString("")
            val byteArray = text.map { byteDecoder[it]!!.toByte() }.toByteArray()
            val decodedText = byteArray.toString(Charsets.UTF_8)
            return decodedText.replace("</w>", " ").trim()
        }

        fun buildInputsWithSpecialTokens(tokenIds0: List<Int>, tokenIds1: List<Int>? = null): IntArray {
            val bosToken = listOf(convertTokenToID(bosToken))
            val eosToken = listOf(convertTokenToID(eosToken))
            var tokens: List<Int>
            if (tokenIds1 == null) {
                tokens = bosToken + tokenIds0 + eosToken
            } else {
                tokens = bosToken + tokenIds0 + eosToken + eosToken + tokenIds1 + eosToken
            }

            return tokens.toIntArray() + IntArray(maxTokens-tokens.size){convertTokenToID(padToken)}
        }

        fun convertTokensToIds(tokens: Any): Any {
            if (tokens == null) {
                return "Error, no tokens supplied"
            }
            return when (tokens) {
                is String -> convertTokenToIdWithAddedVoc(tokens)
                is List<*> -> (tokens as List<String>).map { convertTokenToIdWithAddedVoc(it) }
                else -> throw IllegalArgumentException("Invalid input type")
            }
        }

         fun convertTokenToIdWithAddedVoc(token: String): Int {
            return convertTokenToID(token)
        }
    }
}

