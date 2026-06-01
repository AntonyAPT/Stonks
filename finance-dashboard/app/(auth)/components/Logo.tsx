import Image from 'next/image'

export function Logo() {
  return (
    <div className="flex items-center gap-3 mb-6">
      {/* <div className="w-12 h-12 bg-linear-to-br from-blue-500 to-cyan-600 rounded-lg flex items-center justify-center font-bold text-2xl">
        S
      </div> */}
      <Image src="/Stonks_Logo_White.png" alt="Logo" width={350} height={40} />

    </div>
  );
}
