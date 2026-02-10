import { Github, Linkedin, Mail } from 'lucide-react';

export default function Profile() {
  return (
    <div>
      {/* Header with Avatar and Name */}
      <div className="flex flex-col items-center gap-6 mb-8">
        <div className="relative w-40 h-40">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl blur-3xl opacity-30 animate-pulse" />
          <div className="relative w-40 h-40 rounded-2xl overflow-hidden">
            {/*
              GitHub Pages 使用子路徑（/personal_web_fang/），避免用 /images/... 這種根路徑。
              這裡改用相對路徑，會自動變成 /personal_web_fang/images/avatar.png
            */}
            <img
              src="./images/avatar.png"
              alt="Profile Picture"
              className="object-cover w-full h-full"
              loading="eager"
            />
          </div>
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
            蘇美芳
          </h1>
          <p className="text-xl text-slate-600">Frontend Developer & Backend Developer</p>
        </div>
      </div>

      {/* Contact & Social */}
      <div className="bg-blue-50/80 backdrop-blur-xl p-8 rounded-lg border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-700 mb-4">📬 Contact</h3>

        {/* Social Links */}
        <div className="flex flex-col gap-4">
          <a
            href="https://github.com/hedgehog919?tab=repositories"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 text-slate-600 hover:text-blue-600 transition-colors group"
          >
            <Github size={24} className="group-hover:scale-110 transition-transform" />
            <span>github</span>
          </a>
          <a
            href="https://www.linkedin.com/in/mei-fang-su-697306304/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 text-slate-600 hover:text-blue-600 transition-colors group"
          >
            <Linkedin size={24} className="group-hover:scale-110 transition-transform" />
            <span>linkedin</span>
          </a>
          <a
            href="mailto:sunny45221@gmail.com"
            className="flex items-center gap-3 text-slate-600 hover:text-blue-600 transition-colors group"
          >
            <Mail size={24} className="group-hover:scale-110 transition-transform" />
            <span>gmail</span>
          </a>
        </div>

        {/* Education Section */}
        <div className="mt-8 pt-6 border-t border-blue-200">
          <h4 className="text-lg font-semibold text-blue-700 mb-4">🎓 Education</h4>
          <div className="space-y-4">
            {/* Master's Degree */}
            <div>
              <p className="text-slate-800 font-semibold mb-1">M.S.</p>
              <p className="text-slate-600 text-sm mb-2">NSYSU IMST | 2024 - Present</p>
              {/* <div className="flex gap-4 text-sm">
                <span className="text-blue-600">GPA: 90.50/100</span>
                <span className="text-blue-600">Top 19%</span>
              </div> */}
            </div>

            {/* Bachelor's Degree */}
            <div>
              <p className="text-slate-800 font-semibold mb-1">Junior College</p>
              <p className="text-slate-600 text-sm mb-2">SZMC N | 2016 - 2021</p>
              {/* <div className="flex gap-4 text-sm">
                <span className="text-blue-600">GPA: 89.64/100</span>
                <span className="text-blue-600">Rank: 20/109 (18%)</span>
              </div> */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}