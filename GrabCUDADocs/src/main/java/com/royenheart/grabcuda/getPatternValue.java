package com.royenheart.grabcuda;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 读取html网页中各个元素的属性和值
 */
public class getPatternValue implements get_a {

    public getPatternValue() {
    }

    @Override
    public String get_a_Href(String line) {
        if (line.matches("<a.*</a>")) {
            Pattern href = Pattern.compile("(?<=href=\")[^\"]*(?=\")");
            Matcher m = href.matcher(line);
            if (m.find()) {
                return m.group();
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

    @Override
    public String get_a_InnerHtml(String line) {
        if (line.matches("<a.*</a>")) {
            Pattern innerhtml = Pattern.compile("(?<=<a.{1,999}>).*(?=</a>)");
            Matcher m = innerhtml.matcher(line);
            if (m.find()) {
                return m.group();
            } else {
                return null;
            }
        } else {
            return null;
        }
    }

}
